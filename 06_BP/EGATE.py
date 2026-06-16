import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time

# 시드 고정 (재현성 확보)
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# def generate_random_hamiltonian_graph(N=5):
#     # 현재 RNG state(시드 상태) 백업
#     old_torch_state = torch.random.get_rng_state()
#     old_numpy_state = np.random.get_state()
#     old_py_state = random.getstate()

#     # 그래프만 "진짜 랜덤"으로 만들기 위해, 시드를 None 또는 시스템 시계로 세팅
#     torch.manual_seed(int(time.time()))
    # np.random.seed(int(time.time()))
    # random.seed(int(time.time()))


    
#     # 랜덤 그래프 생성
#     edges = [(i, i+1) for i in range(N-1)]
#     edges.append((N-1, 0))
#     edge_features = {}
#     for e in edges:
#         edge_features[e] = torch.rand(3) * 2.0 - 1.0

    
#     # 그래프 생성 완료 후, 복원
#     torch.random.set_rng_state(old_torch_state)
#     np.random.set_state(old_numpy_state)
#     random.setstate(old_py_state)

#     return edges, edge_features

def one_hot_encoding_nodes(N=5):
    eye = np.eye(N, dtype=np.float32)
    
    return torch.tensor(eye)  # shape (N, N)

# def binary_encoding_nodes(N):
#     num_bits = (N - 1).bit_length() # N개의 노드를 표현할 최소한의 비트 수 결정
#     node_feats = [list(map(int, format(i, f'0{num_bits}b'))) for i in range(N)]
#     return torch.tensor(node_feats, dtype=torch.float)


# ---------------------------------------------------------
# 2. EGATLayer (Node Module + Edge Module) (Node Module에 논문의 λ 적용)
# ---------------------------------------------------------
class EGATLayer(nn.Module):
    def __init__(self,
                 node_in_dim,    # 입력 노드 차원
                 edge_in_dim,    # 입력 엣지 차원
                 node_out_dim,   # 출력 노드 차원
                 edge_out_dim,   # 출력 엣지 차원
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
        
        # --- λ를 통해 노드 부분(F'_H) & 엣지 부분(F'_E) 차원원 결정 ---
        # 예) node_out_dim=16, lambda=0.7 => F'_H=11, F'_E=5
        F_H = int(self.lambda_param * self.node_out_dim)
        F_H = min(F_H, self.node_out_dim)  # just for safety
        F_E = self.node_out_dim - F_H

        self.F_H = F_H  # 노드 부분 차원
        self.F_E = F_E  # 엣지 부분 차원
        # print(F_H)
        # print(F_E)
        # F'_H + F'_E = node_out_dim

        # --- Node Module ---
        # Wh: node_in_dim -> F_H
        # We: edge_in_dim -> F_E
        self.W_node_h = nn.Linear(node_in_dim, F_H, bias=False)  # 노드 파트
        self.W_node_e = nn.Linear(edge_in_dim, F_E, bias=False)  # 엣지 파트

        # 주의: Attention 계산에서 [Wh_i, Wh_j, We_ij]가
        #      dimension = F_H + F_H + F_E = (2F_H + F_E)
        #      = F_H + (F_H + F_E) = F_H + node_out_dim
        self.att_mlp_node = nn.Sequential(
            nn.Linear(F_H + F_H + F_E, hidden_dim_node_att),
            nn.LeakyReLU(), # 기울기 설정 가능 default 0.01
            nn.Linear(hidden_dim_node_att, 1, bias=False)
        )


        # --- Edge Module ---
        # (논문 (7)~(9))은 기존과 동일
        #  여기서는 λ를 적용하지 않고, node_out_dim 전체를 사용
        self.W_h_edge = nn.Linear(node_out_dim, node_out_dim//2, bias=False) # 어텐션을 구할 때는 edge의 영향을 높이기 위해 임의로 node dimension을 낮춤
        self.W_e_edge = nn.Linear(edge_in_dim, node_out_dim, bias=False)

        self.att_mlp_edge = nn.Sequential(
            nn.Linear(node_out_dim + (node_out_dim//2)*2 , hidden_dim_edge_att),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_edge_att, 1, bias=False)
        )
        in_dim_for_edge_mlp = (node_out_dim*2    # h_i, h_j
                               + node_out_dim*2  # e'_i, e'_j
                               + edge_in_dim)     # e_orig
        self.edge_up_mlp = nn.Sequential(
            nn.Linear(in_dim_for_edge_mlp, hidden_dim_edge_update),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_edge_update, edge_out_dim)
        )
    def forward(self, H_in, E_in, edges):
        device = H_in.device
        N = H_in.size(0)
        M = len(edges)

        # 인접 리스트
        adjacency = [[] for _ in range(N)]
        for k, (i, j) in enumerate(edges):
            adjacency[i].append((j, k))
            adjacency[j].append((i, k))

        # -------------------------------------------------
        # [Node Module] (식(2)~(5) + λ 분할)
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
                # 고립 노드의 경우, [Wh_i, 0] 형태로 채움
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
        # [Edge Module] (식(7)~(9) - λ는 적용 안 함)
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
        # E_out = torch.stack(E_out_list, dim=0)
        if len(E_out_list) > 0:
            E_out = torch.stack(E_out_list, dim=0)
        else:
            print("warning!!!")
        return H_out, E_out
# ----------------------- 여기까진 확인

# ---------------------------------------------------------
# 3. MultiLayerEGATEncoder:
#    - Bottleneck -> L개의 EGATLayer -> Merge Layer
# ---------------------------------------------------------
class MultiLayerEGATEncoder(nn.Module):

    def __init__(self,
                 node_in_dim, edge_in_dim,
                 node_hidden_dim, edge_hidden_dim,
                 num_layers=2,
                 lambda_param=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.lambda_param = lambda_param

        # Bottleneck
        # self.bottleneck_node = nn.Linear(node_in_dim, node_hidden_dim)
        # self.bottleneck_edge = nn.Linear(edge_in_dim, edge_hidden_dim)

        # Stacked EGAT Layers
        self.egat_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EGATLayer(
                node_in_dim=node_hidden_dim,
                edge_in_dim=edge_hidden_dim,
                node_out_dim=node_hidden_dim,
                edge_out_dim=edge_hidden_dim,
                lambda_param=self.lambda_param
            )
            self.egat_layers.append(layer)

    def forward(self, H_in, E_in, edges):
        # 1) Bottleneck
        # H0 = self.bottleneck_node(H_in)
        # E0 = self.bottleneck_edge(E_in)

        H = H_in
        E = E_in
        H_list, E_list = [H], [E]

        # 2) stacked EGAT layers
        for layer in self.egat_layers:
            H, E = layer(H, E, edges)
            H_list.append(H)
            E_list.append(E)

        # 3) layer‑wise 평균 (GAT‑v2 안정화 trick)
        H_final = torch.mean(torch.stack(H_list), dim=0)
        E_final = torch.mean(torch.stack(E_list), dim=0)
        # H_final = H_list[-1]
        # E_final = E_list[-1]
        return H_final, E_final
    
# ---------------------------------------------------------
# 4. EGATEAutoEncoder:
#    - Encoder: MultiLayerEGATEncoder
#    - Decoder: graph latent -> edge feature 재구성
# ---------------------------------------------------------
class EGATEAutoEncoder(nn.Module):
    def __init__(self,
                 node_in_dim, edge_in_dim,
                 node_hidden_dim, edge_hidden_dim,
                 num_layers=1,
                 decoder_hidden_dim=32,
                 lambda_param=0.5,
                 num_edges=5):
        super().__init__()
        self.encoder = MultiLayerEGATEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            node_hidden_dim=node_hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=num_layers,
            lambda_param=lambda_param
        )
        self.node_final_dim = node_hidden_dim
        self.edge_final_dim = edge_hidden_dim

        # 본 예시에서는 그래프에 있는 edge 수(M)가 고정이라고 가정
        # (N=5개의 노드 -> 해밀토니안 사이클 = 5개의 에지) 
        # 따라서 디코더는 "graph_latent -> (M * edge_in_dim)" 형태로 매핑하는 구조로 둠.
        
        # num_nodes = 4 #######################################################
        self.num_edges = num_edges
        self.edge_in_dim = edge_in_dim
        self.num_nodes = node_in_dim
        # self.latent_dim = latent_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim  = edge_hidden_dim
        # self._reduce = nn.Linear(node_in_dim * node_hidden_dim + num_edges * edge_hidden_dim, latent_dim)
        # self._expand = nn.Linear(latent_dim, num_nodes * node_hidden_dim + num_edges * edge_hidden_dim)


        # Decoder
        # 입력: graph_latent (node_pool + edge_pool) => (node_hidden_dim + edge_hidden_dim)
        # 출력: M * edge_in_dim (모든 엣지의 특성을 한꺼번에 복원)
        self.decoder = nn.Sequential(
            nn.Linear(self.node_final_dim + self.edge_final_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, node_in_dim * node_hidden_dim + num_edges * edge_hidden_dim)
        )

    def forward(self, H_in, E_in, edges):
        """
        1) EGAT 기반 인코더에서 노드/엣지 임베딩 획득
        2) graph_latent 벡터를 구한 뒤
        3) 해당 벡터만을 입력으로 하여, 모든 엣지 특성 E_in을 한 번에 복원
        """
        N, M = H_in.size(0), E_in.size(0)
        # encode
        H_final, E_final = self.encoder(H_in, E_in, edges)

        # Graph-level latent
        graph_latent = self.get_graph_latent(H_final, E_final)

        # Decode
        flat_re = self.decoder(graph_latent)
        n_dim = H_final.size(1)
        H_out  = flat_re[: N * n_dim]
        E_out  = flat_re[N * n_dim :]
        H_rec = H_out.view(self.num_nodes, -1)  # → (N, node_in_dim)

        # 3) decode edges
          # shape: [M * edge_hidden_dim]
        # E_out = self.edge_decoder(E_lat)  # [M * edge_in_dim]
        E_rec = E_out.view(self.num_edges, -1)  # → (M, edge_in_dim)


        return H_rec, E_rec, graph_latent

    def get_graph_latent(self, H_final, E_final):
        """
        그래프 전체 임베딩:
        노드 임베딩, 엣지 임베딩을 각각 평균 pooling한 뒤 concat
        """
        node_pool = H_final.sum(dim=0)
        edge_pool = E_final.sum(dim=0)
        return torch.cat([node_pool, edge_pool], dim=-1)


# ---------------------------------------------------------
# 5. 학습 루틴/테스트
# ---------------------------------------------------------
def train_single_graph(model, edges, edge_feats_dict, node_feats, num_epochs=1000, stop_threshold=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    mse = nn.MSELoss()

    E_in_list = []
    mse_list = []
    for e in edges:
        E_in_list.append(edge_feats_dict[e])

    # E_in = torch.stack(E_in_list, dim=0)
    if len(E_in_list) > 0:
        E_in = torch.stack(E_in_list, dim=0)
    else:
        raise ValueError("Edge feature list is empty, causing zero-element tensor creation.")
    decay_steps = 50
    decay_rate = 0.65
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: decay_rate ** (step // decay_steps)
    )
    for epoch in range(num_epochs):
        H_rec, E_rec, graph_latent = model(node_feats, E_in, edges)

                # 3) Loss 계산
        E_loss = mse(E_rec, E_in)
        H_loss = mse(H_rec, node_feats)
        loss = E_loss+ H_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_mse = loss.item()
        scheduler.step()
        mse_list.append(current_mse)
        if current_mse < stop_threshold:
            # print(f"Early stopped at epoch={epoch} with MSE={current_mse}")
            break


    with torch.no_grad():
        H_rec ,E_rec, graph_latent = model(node_feats, E_in, edges)
        # final_mse = criterion(E_rec, E_in).item()
        
    return  H_rec, E_rec, mse_list, graph_latent



"""
if __name__ == "__main__":
    N = 5
    node_feats = one_hot_encoding_nodes(N)
    edges, edge_feats_dict = generate_random_hamiltonian_graph(N)

    print("\n=== Training Graph AutoEncoder [EGAT Layer with λ splitting] ===")
    model = EGATEAutoEncoder(
        node_in_dim= N,
        edge_in_dim= 3,
        node_hidden_dim= N,  
        edge_hidden_dim= 3,
        num_layers=3,
        decoder_hidden_dim=16,
        lambda_param=0.5,  # Lambda
        num_edges=len(edges)
    )
    num_epochs = 50
    H_final, E_final, E_rec, mse_list, graph_latent = train_single_graph(
        model, edges, edge_feats_dict, node_feats, num_epochs=num_epochs
    )
    # print(mse_list)
    print(f"[Final MSE] {mse_list[num_epochs-1]:.4f}")
    print("[Encoder - Node Embedding]:", H_final.shape)
    print(H_final)
    print("[Encoder - Edge Embedding]:", E_final.shape)
    print(E_final)
    print("[Graph-level Latent]:", graph_latent.shape)
    print(graph_latent)
    print("[Edge Feature Original vs Reconstructed]")
    for k, e in enumerate(edges):
        print(f"  Edge {e}: org={edge_feats_dict[e]}, rec={E_rec[k]}")

"""