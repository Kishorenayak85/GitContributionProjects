import json


def cite_source(youtube_links_path, retrieved_nodes, auto_merging_response):

    with open(youtube_links_path, 'r') as file:
        youtube_links = json.load(file)

    citations = {}
    if "out of context" not in auto_merging_response.lower():
        for i, node in enumerate(retrieved_nodes):
            file_name = node.metadata['file_name']
            if file_name in youtube_links:
                citation = youtube_links[file_name]
            else:
                citation = file_name.replace(".pdf", "")
            if citation not in citations.values():
                citations[i] = citation
    else:
        pass

    
    return citations