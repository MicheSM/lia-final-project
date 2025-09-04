# Set model to evaluation mode
model.eval()
# Disable gradient computation for evaluation
with torch.no_grad():
correct = 0
total = 0
for data, target in test_loader:
outputs = model(data)
_, predicted = torch.max(outputs.data, 1)
total += target.size(0)
correct += (predicted == target).sum().item()
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')
# Set model back to training mode
model.train()