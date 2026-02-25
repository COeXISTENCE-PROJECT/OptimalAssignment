import torch
import torch.optim as optim


def functional_mlp(x, weights, biases):
    """
    Stateless forward pass of an MLP using explicitly provided
    weights and biases.

    This is used because the policy network parameters
    are generated dynamically by a HyperNetwork.
    """
    for W, b in zip(weights[:-1], biases[:-1]):
        x = torch.relu(x @ W + b)
    return x @ weights[-1] + biases[-1]


def calculate_component2_loss(reconstructed_seq, target_seq, device="cpu"):
    # Proporcjonalnie rosnące wagi dla kroków czasowych
    t_weights = torch.linspace(1.0, 5.0, steps=reconstructed_seq.shape[0]).to(device)

    # MSE dla każdego kroku czasowego uśrednione po węzłach
    mse_per_step = ((reconstructed_seq - target_seq) ** 2).mean(dim=1)

    # Średnia ważona
    return (mse_per_step * t_weights).mean()


def optimize_latent_assignment(
    world_model,
    autoencoder,
    edge_index,
    latent_dim,
    num_nodes,
    device="cpu",
    num_iterations=200,
):
    latent_vector = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optimizer = optim.Adam([latent_vector], lr=0.05)

    world_model.eval()
    autoencoder.eval()

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Rekonstrukcja CAŁEJ sekwencji z latent vector
        reconstructed_seq_flat = autoencoder.decoder(latent_vector)
        reconstructed_seq = reconstructed_seq_flat.view(50, num_nodes)

        # Wybieramy OSTATNI stan (G^T), aby ocenić koszt końcowy
        # (Zgodnie z dokumentacją: optymalizujemy stan końcowy)
        A_final = reconstructed_seq[-1].view(-1, 1)

        # f0(fd(z))
        z_world = world_model.encoder(A_final, edge_index)
        predicted_r = world_model.travel_time_head(z_world.mean(dim=0))

        loss = predicted_r
        loss.backward()
        optimizer.step()

    return latent_vector.detach()


def retrieve_optimal_solution(
    optimized_z, autoencoder, inverse_model, initial_state_A0, no_snaps=50, device="cpu"
):
    # 1. Odzyskujemy pełną sekwencję G^T z wektora latentnego
    optimal_G_T_flat = autoencoder.decoder(optimized_z)
    # Zmieniamy kształt na [no_snaps, num_nodes]
    optimal_G_T = optimal_G_T_flat.view(no_snaps, -1)

    # 2. Problem: autoencoder(initial_state_A0) wybuchnie, bo spodziewa się 50 kroków.
    # Musimy stworzyć "pustą" sekwencję, gdzie tylko pierwszy krok to A0,
    # albo po prostu podać A0 powtórzone 50 razy, aby przejść przez wymiar warstwy Linear.
    dummy_seq = initial_state_A0.repeat(no_snaps, 1)  # [50, 853]
    _, z_start = autoencoder(dummy_seq)

    # 3. Component 1.5: Porównujemy latent stanu początkowego z optymalnym
    optimal_assignment_demand = inverse_model.retrieve_assignment(z_start, optimized_z)

    return optimal_G_T, optimal_assignment_demand
