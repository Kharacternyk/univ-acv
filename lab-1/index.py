import clip
from sklearn.cluster import HDBSCAN
from torch import (
    arange,
    cat,
    cuda,
    empty,
    float32,
    from_numpy,
    inference_mode,
    linalg,
    save,
    split,
)
from torchcodec.decoders import VideoDecoder
from torchvision.datasets import DatasetFolder
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
)
from tqdm import tqdm

device = "cuda" if cuda.is_available() else "cpu"

model, pil_preprocess = clip.load("ViT-B/32", device=device)

if __name__ == "__main__":
    resolution = pil_preprocess.transforms[0].size
    tensor_preprocess = Compose(
        [
            ToDtype(float32, scale=True),
            Resize(resolution, InterpolationMode.BICUBIC),
            CenterCrop(resolution),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    dataset = DatasetFolder(
        "data",
        lambda path: (path, VideoDecoder(path, device=device)),
        extensions=tuple(["webm"]),
    )

    with inference_mode():
        embeddings = empty(0, 512)
        bins = []
        ids = []
        titles = []

        for index, ((path, decoder), _) in tqdm(enumerate(dataset), total=len(dataset)):
            frames = decoder.get_frames_played_at(
                arange(0, decoder.metadata.duration_seconds, 0.5)
            ).data

            features = cat(
                [
                    model.encode_image(tensor_preprocess(frame_batch))
                    for frame_batch in split(frames, 256)
                ]
            )
            features = features / linalg.vector_norm(features, dim=1, keepdim=True)

            cluster_centers = (
                HDBSCAN(
                    metric="cosine",
                    allow_single_cluster=True,
                    store_centers="medoid",
                    min_cluster_size=5,
                )
                .fit(features.cpu().numpy())
                .medoids_
            )

            embeddings = cat([embeddings, from_numpy(cluster_centers)])

            bins += [index] * len(cluster_centers)

            file_name = path.split("/")[-1]
            id = file_name[-17:-6]
            title = file_name[:-19]

            ids.append(id)
            titles.append(title)

        save((embeddings, bins, ids, titles), "index.pt")
