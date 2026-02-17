from clip import tokenize
from streamlit import (
    button,
    cache_resource,
    container,
    markdown,
    session_state,
    text_input,
    toggle,
    video,
)
from torch import inference_mode, linalg, load, ones_like, tensor, topk, zeros

from index import device, model

markdown("# Minuscoogle", text_alignment="center")

query = text_input(
    "пошуковий запит",
    key="query",
    label_visibility="collapsed",
    placeholder="Пошуковий запит…",
).strip()


with container(horizontal=True):
    for example_text in [
        "racing",
        "a yellow toy car",
        "a pink balloon",
        "a rotten apple hanging on a tree branch",
        "potato chips",
        "christmas presents",
        "a purple sock",
        "a white and red checkered box",
    ]:

        def set_query(text=example_text):
            session_state.query = text

        button(example_text, on_click=set_query)

with container(horizontal=True):
    do_prefix = toggle('Додавати "a photo of" до запиту')


@cache_resource
def load_index():
    try:
        return load("index.pt")
    except:
        return load("/assets/index.pt")


embeddings, bins, ids, titles = load_index()
bins = tensor(bins, device=device)
embeddings = embeddings.to(device).float()

if query:
    tokens = tokenize([f"a photo of {query}" if do_prefix else query]).to(device)

    with inference_mode():
        features = model.encode_text(tokens).float()
        features /= linalg.vector_norm(features, dim=1, keepdim=True)

        probabilities = (
            (model.logit_scale.exp() * features @ embeddings.T).squeeze(0).softmax(0)
        )

    binned_probabilities = zeros(len(ids), device=device)
    binned_probabilities.scatter_add_(0, bins, probabilities)

    cluster_counts = zeros(len(ids), device=device)
    cluster_counts.scatter_add_(0, bins, ones_like(probabilities))

    for order_index, score_index in enumerate(topk(binned_probabilities, 5)[1]):
        percentage = int(binned_probabilities[score_index].item() * 100)

        if percentage < 1:
            break

        with container(horizontal=True):
            markdown(
                f"##### {order_index + 1}. {titles[score_index]} "
                f":green-badge[{percentage}%]"
                f":blue-badge[:material/pie_chart_outline: {int(cluster_counts[score_index])}]"
            )

        video(f"https://www.youtube.com/watch?v={ids[score_index]}")
