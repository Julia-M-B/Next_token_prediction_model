import torch
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

checkpoint = torch.load("fined_tuned_model_checkpoint_best.pt")
# checkpoint = torch.load("model_checkpoint_best.pt")
model = checkpoint["model_state_dict"]
epoch = checkpoint["epoch"]
history = checkpoint["history"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]
val_ppl = history["val_ppl"]
# torch.save(model, "fine_tuned_model.pt")

sp_model_path = checkpoint["sp_model_path"]

print(sp_model_path)
print(epoch)

# set style
sns.set_theme(
    context="paper",
    style="whitegrid"
)

colour_1 = "#219CDE"
colour_2 = "#DE6321"

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.plot(range(1, len(train_loss) + 1), train_loss, c=colour_1, label="Zbiór treningowy")
ax.plot(range(1, len(train_loss) + 1), val_loss, c=colour_2,  label="Zbiór walidacyjny")
ax.legend()
ax.set_xticks(range(1, len(train_loss) + 1))
ax.set_xlabel("Liczba epok", fontsize=14, fontweight='bold')
ax.set_ylabel("Wartość funkcji straty", fontsize=14,
              fontweight='bold')
ax.set_title("Funkcja straty modelu w trakcie fine-tuningu", fontsize=16, fontweight='bold')
# ax.set_title("Funkcja straty modelu w trakcie treningu", fontsize=16, fontweight='bold')
plt.savefig("ft_loss.png")

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.plot(range(1, len(train_loss) + 1),val_ppl, c=colour_1)
ax.set_xticks(range(1, len(train_loss) + 1))
ax.set_xlabel("Liczba epok", fontsize=14, fontweight='bold')
ax.set_ylabel("Wartość metryki", fontsize=14,
              fontweight='bold')
ax.set_title("Perpleksja liczona na zbiorze walidacyjnym w trakcie fine-tuningu", fontsize=16, fontweight='bold')
# ax.set_title("Perpleksja liczona na zbiorze walidacyjnym w trakcie treningu", fontsize=16, fontweight='bold')
plt.savefig("ft_ppl.png")