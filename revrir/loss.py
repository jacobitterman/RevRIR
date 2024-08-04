import numpy as np
import torch

def hinge_loss_np(similarity_matrix, embeddings_audio, embeddings_rtf, margin=0.5, alpha = 0.5):
    # Ensure that the similarity matrix and embeddings have the same number of instances
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1] == embeddings_rtf.shape[0] == \
           embeddings_audio.shape[0]
    assert embeddings_rtf.shape[1] == embeddings_audio.shape[1]

    # Calculate hinge loss
    similarity_matrix = np.sign(similarity_matrix)
    dots = np.dot(embeddings_audio, embeddings_rtf.T)
    assert dots.shape == similarity_matrix.shape
    loss_vals = np.maximum(0, margin - similarity_matrix * dots)
    if True:
        pos_loss_vals = loss_vals[similarity_matrix > 0]
        neg_loss_vals = loss_vals[similarity_matrix < 0]
        n_pos = len(pos_loss_vals)
        n_neg = len(neg_loss_vals)
        pos_loss = np.mean(pos_loss_vals)
        neg_loss = np.mean(neg_loss_vals)
        loss = alpha * pos_loss + (1-alpha) * neg_loss
    else:
        loss = loss_vals.mean()
    return loss, pos_loss, neg_loss, n_pos, n_neg

def hinge_loss(similarity_matrix, embeddings_audio, embeddings_rtf,
               margin=0.5, alpha = 0.5, temperature = 0.01, apply_softmax_on_neg = False, softmax_neg_loss_factor = 5):
    # Ensure that the similarity matrix and embeddings have the same number of instances
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1] == embeddings_rtf.shape[0] == \
           embeddings_audio.shape[0]
    assert embeddings_rtf.shape[1] == embeddings_audio.shape[1]

    # Calculate hinge loss
    signed_similarity_matrix = torch.sign(similarity_matrix)
    dots = torch.matmul(embeddings_audio, embeddings_rtf.T)
    assert dots.shape == signed_similarity_matrix.shape
    loss_vals = torch.maximum(torch.tensor(0), margin - signed_similarity_matrix * dots)

    if True:
        pos_loss_vals = loss_vals[signed_similarity_matrix > 0]
        neg_loss_vals = loss_vals[signed_similarity_matrix < 0]
        pos_dots = dots[signed_similarity_matrix > 0]
        neg_dots = dots[signed_similarity_matrix < 0]
        # pos_loss = torch.mean(pos_loss_vals) # TODO: add rank and real accuracy
        if apply_softmax_on_neg:
            pos_loss = torch.mean(pos_loss_vals) # * torch.nn.functional.softmin(pos_dots / temperature, dim=0))
            neg_loss = torch.sum(neg_loss_vals * torch.nn.functional.softmax(neg_dots / temperature, dim=0))
        else:
            pos_loss = torch.mean(pos_loss_vals)
            neg_loss = torch.mean(neg_loss_vals)
        alpha = 0.48
        loss = alpha * pos_loss + (1-alpha) * neg_loss

        print(f"pos_acc: {torch.mean((dots[similarity_matrix == 1] > 0).float()):0.3f}: {torch.sum((dots[similarity_matrix == 1] > 0).int())}/{torch.sum((similarity_matrix == 1).int())}")
        print(
            f"hard_pos_acc: {torch.mean((dots[similarity_matrix == 2] > 0).float()):0.3f}: {torch.sum((dots[similarity_matrix == 2] > 0).int())}/{torch.sum((similarity_matrix == 2).int())}")
        print(
            f"neg_acc: {torch.mean((dots[similarity_matrix == -1] < 0).float()):0.3f}: {torch.sum((dots[similarity_matrix == -1] < 0).int())}/{torch.sum((similarity_matrix == -1).int())}")
        print(
            f"hard_neg_acc: {torch.mean((dots[similarity_matrix == -2] < 0).float()):0.3f}: {torch.sum((dots[similarity_matrix == -2] < 0).int())}/{torch.sum((similarity_matrix == -2).int())}")
    else:
        loss = torch.mean(loss_vals)
    return loss, pos_loss, neg_loss, torch.mean(neg_loss_vals), torch.mean(pos_loss_vals)

def example_run():
    # Example usage

    # Replace this with your actual similarity matrix and embedding vectors
    similarity_matrix = np.array([[2, -1, 2], [-1, 1, -1], [1, -1, 1]])
    audio_embeddings = np.random.rand(3, 50)  # 100 samples, each with a 50-dimensional embedding
    rtf_embeddings = np.random.rand(3, 50)  # 100 samples, each with a 50-dimensional embedding

    # Calculate hinge loss
    loss_np = hinge_loss_np(similarity_matrix, audio_embeddings, rtf_embeddings)[0]

    similarity_matrix_torch = torch.tensor(similarity_matrix, dtype=torch.int32)
    audio_embeddings_torch = torch.tensor(audio_embeddings, dtype=torch.float32)
    rtf_embeddings_torch = torch.tensor(rtf_embeddings, dtype=torch.float32)
    loss_torch = hinge_loss(similarity_matrix_torch, audio_embeddings_torch, rtf_embeddings_torch)[0]
    print("Hinge Loss np:   ", loss_np)
    print("Hinge Loss Torch:", loss_torch)
    assert loss_np - loss_torch.cpu().numpy() < 1E-5
    print('test passed')

    from IPython import embed; embed()


if __name__ == '__main__':
    example_run()