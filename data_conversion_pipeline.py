from datasets import load_dataset
import networkx as nx
import tqdm
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from transformers import pipeline

webNLG_dataset = load_dataset("web_nlg", "webnlg_challenge_2017")

distributed_state = PartialState()
pipe = pipeline("text-to-speech", "suno/bark", device=distributed_state.device)

## Converting tripple set to a graph
def generate_graph_for_tripple_set(tripple_set):
    G = nx.Graph()
    for tripple in tripple_set:
        tripple = tripple.split(" | ")
        ## check if node is in G
        if tripple[0] not in G.nodes:
            G.add_node(tripple[0])
        if tripple[1] not in G.nodes:
            G.add_node(tripple[1])
        if tripple[2] not in G.nodes:
            G.add_node(tripple[2])
        ## Add edge between nodes
        G.add_edge(tripple[0], tripple[1])
        G.add_edge(tripple[1], tripple[2])
    return nx.cytoscape_data(G)

## Printing function for each
def print_details(idx):
    print("EID: ", eid_list[idx])
    for idx, text in enumerate(text_list[idx]):
        print(f"Option {idx}: {text}\n")
    
    print("Graph Visual:")
    nx.draw(nx.cytoscape_graph(graph_list[idx]),with_labels=True, font_size=8)
    
#def generate_audio_content(text):
#    outputs = synthesizer([text])
#    return outputs[0]["audio"]

G = generate_graph_for_tripple_set(webNLG_dataset["train"][0]["modified_triple_sets"]["mtriple_set"][0])

## Iterate through all the triple sets and generate graphs
eid_list = []
graph_list = []
text_list = []
wav_list = []

for each in tqdm.tqdm(webNLG_dataset["train"]):
    eid_list.append(each["eid"])
    text_list.append(each["lex"]["text"])
    ## Create the WAV file:
#    each_wav_list = []
#    for each_text in each["lex"]["text"]:
#       each_wav_list.append(generate_audio_content(each_text))
#   wav_list.append(each_wav_list)
    G = generate_graph_for_tripple_set(each["modified_triple_sets"]["mtriple_set"][0])
    graph_list.append(G)

## Save the data
import pandas as pd
df = pd.DataFrame({
    "eid": eid_list,
    "graph": graph_list,
    "text": text_list,
})

df = df.explode('text').reset_index(drop=True)

with distributed_state.split_between_processes(list(df["text"].values)) as prompt:
    result = pipe(prompt)

df["wav"] = result

from datasets import Dataset
processed_dataset = Dataset.from_pandas(df)

## Save the processed dataset
processed_dataset.save_to_disk("/home/CS546-CLEO/data/processed_dataset.hf")