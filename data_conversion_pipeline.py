from datasets import load_dataset
import networkx as nx
import tqdm
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from transformers import pipeline
import uuid
import os
import pandas as pd
import pickle
from datasets import Dataset

webNLG_dataset = load_dataset("web_nlg", "webnlg_challenge_2017")

distributed_state = PartialState()
pipe = pipeline("text-to-speech", "suno/bark", device=distributed_state.device, batch_size=16)

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

def load_all_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

## Printing function for each
def print_details(idx):
    print("EID: ", eid_list[idx])
    for idx, text in enumerate(text_list[idx]):
        print(f"Option {idx}: {text}\n")
    
    print("Graph Visual:")
    nx.draw(nx.cytoscape_graph(graph_list[idx]),with_labels=True, font_size=8)
    
G = generate_graph_for_tripple_set(webNLG_dataset["train"][0]["modified_triple_sets"]["mtriple_set"][0])

## Iterate through all the triple sets and generate graphs
eid_list = []
graph_list = []
text_list = []
wav_list = []

for each in tqdm.tqdm(webNLG_dataset["train"]):
    eid_list.append(each["eid"])
    text_list.append(each["lex"]["text"])
    G = generate_graph_for_tripple_set(each["modified_triple_sets"]["mtriple_set"][0])
    graph_list.append(G)

## Save the data
df = pd.DataFrame({
    "eid": eid_list,
    "graph": graph_list,
    "text": text_list,
})

df = df.explode('text').reset_index(drop=True)
df = df.head(400)

with distributed_state.split_between_processes(list(df["text"].values)) as prompt:
    store_file = {
        "wav": pipe(prompt),
        "text": prompt
    }
    with open(f"/home/CS546-CLEO/data/tmp/{str(uuid.uuid4()).replace('-','')}", "wb") as f:
        pickle.dump(store_file,f)
    

tmp_files = load_all_files_in_folder("data/tmp")
## Create the audio to text mapping dataframe
import pickle
audio_list = []
txt_list = []
for each_file in tmp_files:
    with open(each_file, "rb") as f:
        tmp_dict = pickle.load(f)
        audio_list.extend(tmp_dict["wav"])
        txt_list.extend(tmp_dict["text"])

audio_df = pd.DataFrame({"wav": audio_list, "text": txt_list})

## Merge the two dataframes
df = df.merge(audio_df, on="text", how="inner")

## Remove the tmp audio files
for each_file in tmp_files:
    os.remove(each_file)

## Create the HF dataset
processed_dataset = Dataset.from_pandas(df)

## Save the processed dataset
processed_dataset.save_to_disk("/home/CS546-CLEO/data/processed_dataset")