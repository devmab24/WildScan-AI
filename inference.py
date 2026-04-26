def predict(model, tensor, classes):
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]