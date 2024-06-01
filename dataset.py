import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DashcamDataset(Dataset):
    def __init__(self, video_folder, annotations, transform=None):
        self.video_folder = video_folder
        self.annotations = annotations  # A list of dictionaries containing location and count information
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_path = os.path.join(self.video_folder, annotation['video_name'])
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()
        
        location = annotation['location']
        vehicle_count = annotation['vehicle_count']
        truck_count = annotation['truck_count']
        pedestrian_count = annotation['pedestrian_count']
        
        counts = torch.tensor([vehicle_count, truck_count, pedestrian_count], dtype=torch.float32)
        return torch.stack(frames), location, counts

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

annotations = [
    {'video_name': 'video1.mp4', 'location': [1, 0, 0], 'vehicle_count': 5, 'truck_count': 2, 'pedestrian_count': 3},
    # Add more annotations here
]

def get_dataloader(video_folder, annotations, batch_size=2, shuffle=True):
    dataset = DashcamDataset(video_folder=video_folder, annotations=annotations, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
