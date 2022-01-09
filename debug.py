import torch
import torchaudio

for epoch in range(10):
  print(f"Epoch {epoch+1}/{30}")
  for example in range(20000):
    if (example + 1) % 100 == 0:
      print(f"Example {example + 1}")
    x = torch.rand(1, 88200).cuda()
    mel_spec = torchaudio.transforms.MelSpectrogram().cuda()(x)
