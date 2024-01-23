import networkx as nx

def load_edge_from_txt(file_name,edgelist,splt):    
    with open(file_name,'r') as f:        
        for line in f: 
            edgelist.append(list(map(int,line.strip('\n').split(splt))))
def get_graph_from_txt(path,splt):
    edges = []
    load_edge_from_txt(path,edges,splt)
    return nx.from_edgelist(edges)
def get_lcc_subgraph_from_txt(path,splt):
    G_o = get_graph_from_txt(path,splt)
    largest_cc = max(nx.connected_components(G_o), key=len)
    G = G_o.subgraph(largest_cc).copy() 
    return G