def train_loop(model, X, y, mask, criterion, optimizer, epochs):
    for ep in range(epochs):
        print(f"Epoch: {ep + 1}")
        preds = model.forward(X)
        preds = preds * mask
        loss = criterion(preds, y)
        print(f"Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()