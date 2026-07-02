import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from typing import List, Tuple

# Fix seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


def one_hot_encoding_nodes(N=5):
    eye = np.eye(N, dtype=np.float32)
    
    return torch.tensor(eye)  # shape (N, N)

def one_hot_encoding_nodes_single_term(N=5, K = 0):
    M = torch.cat([torch.eye(N, dtype=torch.float32),
               torch.full((N, 1), K)], dim=1)

    return M  # shape (N, N)

def one_encoding_nodes_single_term(N=5, K = 0):
    M = torch.cat([torch.full((N, 1), 1),
               torch.full((N, 1), K)], dim=1)

    return M  # shape (N, N)




# ---------------------------------------------------------
# 2. EGATLayer (Node Module + Edge Module; apply the paper lambda split to the node module)
# ---------------------------------------------------------
class EGATLayer(nn.Module):
    def __init__(self,
                 node_in_dim,    # input node dimension
                 edge_in_dim,    # input edge dimension
                 node_out_dim,   # output node dimension
                 edge_out_dim,   # output edge dimension
                 lambda_param=0.5,
                 hidden_dim_node_att=16,
                 hidden_dim_edge_att=16,
                 hidden_dim_edge_update=32):
        super().__init__()
        
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim
        self.lambda_param = lambda_param
        
        # --- Use lambda to split dimensions into node part (F'_H) and edge part (F'_E) ---
        # Example: node_out_dim=16, lambda=0.7 => F'_H=11, F'_E=5
        F_H = int(self.lambda_param * self.node_out_dim)
        F_H = min(F_H, self.node_out_dim)  # just for safety
        F_E = self.node_out_dim - F_H

        self.F_H = F_H  # node-part dimension
        self.F_E = F_E  # edge-part dimension
        # F'_H + F'_E = node_out_dim

        # --- Node Module ---
        # Wh: node_in_dim -> F_H
        # We: edge_in_dim -> F_E
        self.W_node_h = nn.Linear(node_in_dim, F_H, bias=False)  # node part
        self.W_node_e = nn.Linear(edge_in_dim, F_E, bias=False)  # edge part

        # Note: in attention computation, [Wh_i, Wh_j, We_ij]
        #      dimension = F_H + F_H + F_E = (2F_H + F_E)
        #      = F_H + (F_H + F_E) = F_H + node_out_dim
        self.att_mlp_node = nn.Sequential(
            nn.Linear(F_H + F_H + F_E, hidden_dim_node_att),
            nn.LeakyReLU(), # configurable slope, default 0.01
            nn.Linear(hidden_dim_node_att, hidden_dim_node_att//4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_node_att//4, 1, bias=False)
        )

        # --- Edge Module ---
        #  Do not apply lambda here; use the full node_out_dim
        self.W_h_edge = nn.Linear(node_out_dim, node_out_dim//2, bias=False) # Reduce node dimension to emphasize edge effects in attention
        self.W_e_edge = nn.Linear(edge_in_dim, node_out_dim, bias=False)

        self.att_mlp_edge = nn.Sequential(
            nn.Linear(node_out_dim + (node_out_dim//2)*2 , hidden_dim_edge_att),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_edge_att , hidden_dim_edge_att//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_edge_att//2, 1, bias=False)
        )
        in_dim_for_edge_mlp = (node_out_dim*2    # h_i, h_j
                               + node_out_dim*2  # e'_i, e'_j
                               + edge_in_dim)     # e_orig
        self.edge_up_mlp = nn.Sequential(
            nn.Linear(in_dim_for_edge_mlp, hidden_dim_edge_update),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_edge_update, hidden_dim_edge_update//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_edge_update//2, edge_out_dim)
        )
    def reset_parameters(self):
            """
            Apply LeakyReLU/Kaiming initialization to all Linear layers.
            Set bias to zero.
            """
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight,
                                        a=0.01,          # LeakyReLU slope
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    def forward(self, H_in, E_in, edges):
        device = H_in.device
        N = H_in.size(0)
        M = len(edges)

        # adjacency list
        adjacency = [[] for _ in range(N)]
        for k, (i, j) in enumerate(edges):
            adjacency[i].append((j, k))
            adjacency[j].append((i, k))

        # -------------------------------------------------
        # [Node Module] 
        # -------------------------------------------------
        Wh = self.W_node_h(H_in) # Wh(h): (N, F_H)
        We = self.W_node_e(E_in) # We(e): (M, F_E)

        if N > 0:
            H_out = torch.zeros((N, self.node_out_dim), device=device)
        else:
            raise ValueError("N is zero, which leads to an empty tensor.")
        
        for i in range(N):
            neighbors = adjacency[i]
            if len(neighbors) == 0:
                # For isolated nodes, fill with [Wh_i, 0].
                #   Wh_i.shape=(F_H), 0.shape=(F_E)
                Wh_i = Wh[i]
                pad_e = torch.zeros(self.F_E, device=device)
                H_out[i] = torch.cat([Wh_i, pad_e], dim=-1)
                continue

            scores = []
            neighbor_feats = []

            Wh_i = Wh[i]  # (F_H)
            for (j, k_edge) in neighbors:
                Wh_j = Wh[j]         # (F_H)
                We_ij = We[k_edge]   # (F_E)

                # Attention score => [Wh_i, Wh_j, We_ij] => (F_H + F_H + F_E)
                concat_ij = torch.cat([Wh_i, Wh_j, We_ij], dim=-1)
                score_ij = self.att_mlp_node(concat_ij)
                scores.append(score_ij)

                # aggregator => [Wh_j || We_ij] => (F_H + F_E) => node_out_dim
                neighbor_feats.append(torch.cat([Wh_j, We_ij], dim=-1))

            # scores = torch.stack(scores, dim=0)  # (num_neighbors,1)
            # alpha = F.softmax(scores, dim=0)     # (num_neighbors,1)
            
            if len(scores) > 0:
                scores = torch.stack(scores, dim=0)  # (num_neighbors,1)
                alpha = F.softmax(scores, dim=0)     # (num_neighbors,1)
            else:
                scores = torch.tensor([], device=device)
                alpha = torch.tensor([], device=device)

            neighbor_feats = torch.stack(neighbor_feats, dim=0)  # (num_neighbors, node_out_dim)

            # h'_i = sum_j alpha_ij * [Wh_j || We_ij]
            h_new_i = (alpha * neighbor_feats).sum(dim=0)
            h_new_i = F.elu(h_new_i)
            H_out[i] = h_new_i

        # -------------------------------------------------
        # [Edge Module] 
        # -------------------------------------------------
        Wh_edge = self.W_h_edge(H_out)  # (N, node_out_dim//2)
        We_edge = self.W_e_edge(E_in)   # (M, node_out_dim)

        e_i_agg = torch.zeros((N, self.node_out_dim), device=device)

        for i in range(N):
            neighbors = adjacency[i]
            if len(neighbors) == 0:
                continue
            scores_edge = []
            edge_vecs = []
            hi_ = Wh_edge[i]
            for (j, k_edge) in neighbors:
                hj_ = Wh_edge[j]
                e_ij_ = We_edge[k_edge]
                concat_ij = torch.cat([hi_, hj_, e_ij_], dim=-1)  # (3*node_out_dim)
                score_ij = self.att_mlp_edge(concat_ij)
                scores_edge.append(score_ij)
                edge_vecs.append(e_ij_)

            scores_edge = torch.stack(scores_edge, dim=0)
            beta = F.softmax(scores_edge, dim=0)
            edge_vecs = torch.stack(edge_vecs, dim=0)
            e_i_new = (beta * edge_vecs).sum(dim=0)
            e_i_agg[i] = e_i_new

        E_out_list = []
        for k, (i, j) in enumerate(edges):
            hi = H_out[i]
            hj = H_out[j]
            ei_agg = e_i_agg[i]
            ej_agg = e_i_agg[j]
            e_orig = E_in[k]
            x_ij = torch.cat([hi, hj, ei_agg, ej_agg, e_orig], dim=-1)
            e_ij_new = self.edge_up_mlp(x_ij)
            E_out_list.append(e_ij_new)
        if len(E_out_list) > 0:
            E_out = torch.stack(E_out_list, dim=0)
        else:
            print("warning!!!")
        return H_out, E_out

# ---------------------------------------------------------
# 3. MultiLayerEGATEncoder:
#    - Bottleneck -> L EGATLayer -> Merge Layer
# ---------------------------------------------------------
class MultiLayerEGATEncoder(nn.Module):

    def __init__(self,
                node_in_dim: int,
                edge_in_dim: int,
                node_hidden_dim: int,
                edge_hidden_dim: int,
                num_layers: int = 2,
                lambda_param: float = 0.5):
        super().__init__()
        
        # Stacked EGAT Layers
        self.egat_layers = nn.ModuleList(
            [
                EGATLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    node_hidden_dim,
                    edge_hidden_dim,
                    lambda_param,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, H_in: torch.Tensor, E_in: torch.Tensor, edges):
        # 1) Bottleneck
        # H0 = self.bottleneck_node(H_in)
        # E0 = self.bottleneck_edge(E_in)

        H = H_in
        E = E_in
        H_list, E_list = [], []

        # 2) stacked EGAT layers
        for layer in self.egat_layers:
            H, E = layer(H, E, edges)
            H_list.append(H)
            E_list.append(E)

        # 3) Layer-wise averaging (GAT-v2 stabilization trick)
        H_final = torch.mean(torch.stack(H_list), dim=0)
        E_final = torch.mean(torch.stack(E_list), dim=0)
        return H_final, E_final
    
# ---------------------------------------------------------
# 4. EGATEAutoEncoder:
#    - Encoder: MultiLayerEGATEncoder
#    - Decoder: graph latent -> edge feature reconstruction
# ---------------------------------------------------------
class EGATEAutoEncoder(nn.Module):
    def __init__(self,
                node_in_dim: int,
                edge_in_dim: int,
                node_hidden_dim: int,
                edge_hidden_dim: int,
                decoder_hidden_dim:int,
                num_layers: int = 1,
                lambda_param: float = 0.5,
                latent_dim: int = 18,
                num_edges: int = 6 
                ):
        super().__init__()
        
        self.encoder = MultiLayerEGATEncoder(node_in_dim,
                                            edge_in_dim,
                                            node_hidden_dim,
                                            edge_hidden_dim,
                                            num_layers,
                                            lambda_param
                                            )
        # self.node_hidden_dim = node_hidden_dim
        # self.edge_hidden_dim = edge_hidden_dim
        self.node_final_dim = node_hidden_dim
        self.edge_final_dim = edge_hidden_dim

        # This implementation assumes a fixed number of graph edges (M).
        # (N=5 nodes -> Hamiltonian cycle with 5 edges) 
        # The decoder maps graph_latent to M * edge_in_dim.
        
        num_nodes = num_layers-1      
        self.num_edges = num_edges
        self.edge_in_dim = edge_in_dim
        self.num_nodes = node_in_dim
        # self.latent_dim = latent_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim  = edge_hidden_dim
        self._reduce = nn.Linear(num_layers*(node_in_dim * node_hidden_dim + num_edges * edge_hidden_dim), latent_dim)
        # self._expand = nn.Linear(latent_dim, num_nodes * node_hidden_dim + num_edges * edge_hidden_dim)


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, num_nodes * node_hidden_dim + num_edges * edge_hidden_dim)
        )

    def forward(self, H_in, E_in, edges):
        # encode

        N, M = H_in.size(0), E_in.size(0)
        n_dim = H_in.size(1)
        H_final, E_final = self.encoder(H_in, E_in, edges)

        # Graph-level latent
        graph_latent = self.get_graph_latent(H_final, E_final)
        flat_re = self.decoder(graph_latent)

        # Decode
        # flat_re = self.decoder(graph_latent)
        
        H_out  = flat_re[: N * n_dim]
        E_out  = flat_re[N * n_dim :]
        H_rec = H_out.view(N, -1)  # → (N, node_in_dim)
        # 3) decode edges
          # shape: [M * edge_hidden_dim]
        E_rec = E_out.view(self.num_edges, -1)  # → (M, edge_in_dim)
        return H_rec, E_rec, graph_latent
    
    def get_graph_latent(self, H_final, E_final):
        """
        Graph-level embedding:
        Mean-pool node and edge embeddings separately, then concatenate.
        """
        # node_pool = H_final.sum(dim=0)
        # edge_pool = E_final.sum(dim=0)
        # return torch.cat([H_final.flatten(), E_final.flatten()])
        node_pool = H_final.sum(dim=0)
        edge_pool = E_final.sum(dim=0)
        return torch.cat([node_pool, edge_pool], dim=-1)


# ---------------------------------------------------------
# 5. training routine / test
# ---------------------------------------------------------
def train_multi_graph_minibatch(
        model, num_data, edges, edge_full_data, node_full_data, 
                      num_epochs=100, lr=0.001,batch_size: int = 50,
                      beta: float = 6.0, stop_threshold=1e-5,shuffle: bool = True,):

    """
    Train with minibatches (default 50). This works even when the model only supports graph-level forward passes.
    Average losses across graphs in the batch, then call backward() once.
    """
    device = next(model.parameters()).device
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    decay_steps = 10
    decay_rate = 0.8
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: decay_rate ** (step // decay_steps)
    )

    mse = nn.MSELoss()
    mse_list_per_epoch = []
    H_mse_list = []
    E_mse_list = []

    indices = np.arange(num_data)

    for epoch in range(num_epochs):
        if shuffle:
            np.random.shuffle(indices)

        total_E_loss = 0.0
        total_H_loss = 0.0

        # ----- minibatch loop -----
        for start in range(0, num_data, batch_size):
            end = min(start + batch_size, num_data)
            batch_idx = indices[start:end]

            optimizer.zero_grad()
            batch_loss_tensor = 0.0
            batch_E_loss_sum = 0.0
            batch_H_loss_sum = 0.0

            for i in batch_idx:
                # 1) Build inputs
                node_feats = node_full_data[i].to(device)                     # assume (N_nodes, d_node)
                edge_feats_dict = edge_full_data[i]                           # dict[(u,v)] -> tensor(d_edge,)
                E_in_list = [edge_feats_dict[e] for e in edges]
                E_in = torch.stack(E_in_list, dim=0).to(device)               # (num_edges, d_edge)

                # 2) Forward
                H_rec, E_rec, _ = model(node_feats, E_in, edges)

                # 3) Loss
                E_loss = mse(E_rec, E_in)
                H_loss = mse(H_rec, node_feats)
                loss_i = E_loss + H_loss

                batch_loss_tensor = batch_loss_tensor + loss_i
                batch_E_loss_sum += E_loss.item()
                batch_H_loss_sum += H_loss.item()

            # Stabilize scale with batch-averaged loss
            batch_size_eff = len(batch_idx)
            (batch_loss_tensor / batch_size_eff).backward()
            optimizer.step()

            # Accumulate epoch total to compute the per-sample average
            total_E_loss += batch_E_loss_sum
            total_H_loss += batch_H_loss_sum

        # ----- epoch-end handling -----
        avg_E = total_E_loss / num_data
        avg_H = total_H_loss / num_data
        avg_total = (total_E_loss + total_H_loss) / num_data

        mse_list_per_epoch.append(avg_total)
        H_mse_list.append(avg_H)
        E_mse_list.append(avg_E)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss={avg_total:.6f}  (H={avg_H:.6f}, E={avg_E:.6f})")

        if avg_total < stop_threshold:
            print(f"Early stopped at epoch={epoch} with MSE={avg_total:.6e}")
            break

        scheduler.step()

    # -------- After training: extract final latent vectors/reconstructions for each graph --------
    model.eval()
    latents = []
    H_rec_list = []
    E_rec_list = []
    with torch.no_grad():
        for i in range(num_data):
            node_feats = node_full_data[i].to(device)
            edge_feats_dict = edge_full_data[i]
            E_in_list = [edge_feats_dict[e] for e in edges]
            E_in = torch.stack(E_in_list, dim=0).to(device)

            H_re, E_re, latent = model(node_feats, E_in, edges)
            latents.append(latent.detach().cpu())
            H_rec_list.append(H_re.detach().cpu())
            E_rec_list.append(E_re.detach().cpu())

    return latents, mse_list_per_epoch, H_rec_list, E_rec_list, H_mse_list, E_mse_list
