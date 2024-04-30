from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


def calculate_iou(bbox1, bbox2) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1 = max(x1, x2)
    y1 = max(y1, y2)
    x2 = min(x1 + w1, x2 + w2)
    y2 = min(y1 + h1, y2 + h2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (w1 * h1) + (w2 * h2) - intersection
    return intersection / union
    

def eval_model(model, dataset, transforms, device) -> list[float]:
    model.eval()
    ious = []
    for sample in tqdm(dataset):
        img = Image.open(sample.filepath)
        img = np.array(img)
        img = transforms(img)
        img = torch.tensor(img).float().unsqueeze(0)
        img = img.to(device)
        preds = model(img)
        preds = preds.cpu().detach().numpy().squeeze()

        bbox = sample.detections.detections[0].bounding_box
        top_left_x, top_left_y, width, height = np.array(bbox) * img.shape[-1]

        x1, y1, x2, y2 = np.array(preds)

        iou = calculate_iou((top_left_x, top_left_y, width, height), (x1, y1, x2-x1, y2-y1))
        ious.append(iou)
    print(f"Mean IoU: {np.mean(ious)}")
    return ious