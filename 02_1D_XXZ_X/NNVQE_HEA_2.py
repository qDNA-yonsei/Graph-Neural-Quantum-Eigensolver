from qiskit import *
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit.primitives import StatevectorEstimator 
import torch
import numpy as np

def Energy(circuit, edges, edge_data, node_feats, compute_state=False):
    """
    circuit: Qiskit 회로 (QuantumCircuit)
    edges:   [(i, j), (k, l), ...] 형태의 엣지 리스트
    edge_data: 각 엣지별 [J_xx, J_yy, J_zz] 값을 담은 리스트
               edges와 동일한 순서로 정렬되어 있어야 함
    """
    n = circuit.num_qubits  # 회로에 사용되는 큐빗 수
    coeffs = []
    paulis = []
    K_x = node_feats[0][-1].item()

    # 주어진 edges와 edge_data를 함께 순회
    for (q1, q2), (J_xx, J_yy, J_zz) in zip(edges, edge_data):
        # X_q1 X_q2
        pauli_string = ['I'] * n
        pauli_string[q1] = 'X'
        pauli_string[q2] = 'X'
        paulis.append(Pauli(''.join(pauli_string)))
        coeffs.append(J_xx)

        # Y_q1 Y_q2
        pauli_string = ['I'] * n
        pauli_string[q1] = 'Y'
        pauli_string[q2] = 'Y'
        paulis.append(Pauli(''.join(pauli_string)))
        coeffs.append(J_yy)

        # Z_q1 Z_q2
        pauli_string = ['I'] * n
        pauli_string[q1] = 'Z'
        pauli_string[q2] = 'Z'
        paulis.append(Pauli(''.join(pauli_string)))
        coeffs.append(J_zz)

    for i in range(n):
        s = ['I'] * n
        s[i] = 'X'
        paulis.append(Pauli(''.join(s)))
        coeffs.append(K_x)


    # 해밀토니안 구성
    H = SparsePauliOp(paulis, coeffs=coeffs)

    # StatevectorEstimator를 이용해 기대값 계산
    estimator = StatevectorEstimator()
    if compute_state:
        state = Statevector.from_instruction(circuit)
        state_array = state.data

        job = estimator.run([(circuit, H)])  # [(회로, 해밀토니안)] 형태
        result = job.result()
        energy = result[0].data.evs  # 기대값(에너지)

        return energy, state_array
    else:
        job = estimator.run([(circuit, H)])  # [(회로, 해밀토니안)] 형태
        result = job.result()
        energy = result[0].data.evs  # 기대값(에너지)
        state_array = [0]
    return energy, state_array



    
def HEA(inp, n, d=1, energy_flag=False, param_num=False):
    params = inp["params"]
    edges = inp["edges"]
    edge_data = inp["edge_data"]
    node_feats = inp["node_feats"]
    qc = QuantumCircuit(n)

    idx = 0

    for i in range(n):
        qc.rx(params[3 * i],i)
        qc.rz(params[3 * i + 1],i)
        qc.rx(params[3 * i + 2],i)
    idx += 3 * n

    for _ in range(d):
        for i in range(0, n):
            qc.rzz(params[idx], i, (i + 1) % n)
            idx += 1

        for i in range(0, n):
            qc.rxx(params[idx], i, (i + 1) % n)
            idx += 1

        for i in range(0, n):
            qc.ryy(params[idx], i, (i + 1) % n)
            idx += 1

        for i in range(n):
            qc.rx(params[idx],i)
            qc.rz(params[idx + 1],i)
            idx += 2

    if energy_flag:
        e, _ = Energy(qc, edges, edge_data, node_feats)
        return e
    elif param_num:
        return qc, idx
    else:
        return qc

def compute_gradients(params_np, edges, edge_data, node_feats, n, d):
    shift = np.pi / 2
    num_params = len(params_np)
    grad_params = np.zeros(num_params)

    for i in range(num_params):
        shifted_params_plus = np.copy(params_np)
        shifted_params_minus = np.copy(params_np)
        shifted_params_plus[i] += shift
        shifted_params_minus[i] -= shift

        # 양자 회로 생성
        qc_plus = HEA({"params": shifted_params_plus, "edges": edges, "edge_data": edge_data, "node_feats": node_feats}, n, d)
        qc_minus = HEA({"params": shifted_params_minus, "edges": edges, "edge_data": edge_data, "node_feats": node_feats}, n, d)

        # 에너지 계산
        energy_plus, _ = Energy(qc_plus, edges, edge_data, node_feats)
        energy_minus, _ = Energy(qc_minus, edges, edge_data, node_feats)

        # 기울기 계산
        grad_params[i] = 0.5 * (energy_plus - energy_minus)

    return grad_params

class QuantumCircuitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, edges, edge_data, node_feats, n, d, compute_state):
        ctx.save_for_backward(params)
        ctx.edges = edges
        ctx.edge_data = edge_data
        ctx.node_feats = node_feats
        ctx.n = n
        ctx.d = d

        params_np = params.detach().cpu().numpy()

        qc = HEA({"params": params_np, "edges": edges, "edge_data": edge_data, "node_feats": node_feats}, n, d)
        energy, state = Energy(qc, edges, edge_data, node_feats, compute_state)

        return torch.tensor(energy, dtype=params.dtype), torch.tensor(state, dtype=torch.complex128)

    @staticmethod
    def backward(ctx, grad_output, grad_output_state):
        params, = ctx.saved_tensors
        edges     = ctx.edges
        edge_data = ctx.edge_data
        node_feats = ctx.node_feats
        n = ctx.n
        d = ctx.d

        params_np = params.detach().cpu().numpy()

        grad_params_np = compute_gradients(params_np, edges, edge_data, node_feats, n, d)
        grad_params = torch.from_numpy(grad_params_np).to(params.device).float()

        return grad_output * grad_params, None, None, None, None, None, None

class NN_MERA_Model(torch.nn.Module):
    def __init__(self, n, d, stddev, NN_shape, latent_size, dropout_rate):
        super(NN_MERA_Model, self).__init__()
        self.n = n
        self.d = d

        # MERA에 필요한 파라미터 수 계산
        _, idx = HEA({"params": np.zeros(1000), "edges": None,
            "edge_data": None, "node_feats":None}, n, d, param_num=True)
        self.idx = idx

        # 신경망 레이어 정의
        # self.input_layer = torch.nn.Linear(1, 20)  # 입력 차원은 상황에 따라 조정
        # torch.nn.init.normal_(self.input_layer.weight, mean=0.0, std=stddev)
        self.hidden_layer1 = torch.nn.Linear(latent_size, NN_shape)
        torch.nn.init.normal_(self.hidden_layer1.weight, mean=0.0, std=stddev)
        
        self.hidden_layer2 = torch.nn.Linear(NN_shape, NN_shape*2)
        torch.nn.init.normal_(self.hidden_layer2.weight, mean=0.0, std=stddev)
        
        self.output_layer = torch.nn.Linear(NN_shape*2, self.idx)
        torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=stddev)

        # 드롭아웃 레이어 정의
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self,edges, edge_data, node_feats, latent_vector, compute_state=False):
        """
        latent_vector: shape (batch, t) 혹은 (t,) 형태의 텐서
        """
        

        # (1) 배치 차원 처리 (예: batch=1로 가정)
        #     여기서는 간단히 [t] -> [1,t] 형태 맞추거나, 이미 [B,t]이면 그대로 사용
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)  # [t] -> [1, t]
        # print(latent_vector)
        # (2) 신경망 통과
        x = self.hidden_layer1(latent_vector)
        x = torch.relu(x)
        x = self.hidden_layer2(x)
        x = torch.relu(x)
        # x = self.dropout(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        # (3) 파라미터 스케일링
        params = x * 6.3  # 원하는 스케일로 조정

        # (4) 배치 여러 개라면 loop / mean / sum 등의 처리 필요
        #     여기서는 batch=1 가정 -> (1, num_params) -> (num_params,)
        params = params.view(-1)

        # (5) Custom Autograd Function을 사용해 에너지 계산
        energy, state = QuantumCircuitFunction.apply(params, edges, edge_data,node_feats, self.n, self.d, compute_state)
        # print(state)
        return energy, state

  
