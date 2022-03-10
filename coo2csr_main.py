from ogb.nodeproppred import PygNodePropPredDataset
papers_path = "/hdd1/datasets_gujy/ogb_datasets/dataset/"
save_path = "/hdd1/datasets_gujy/ogb_datasets/CSR/"
dataset = PygNodePropPredDataset(name = 'ogbn-papers100M',root=papers_path) ## please set the root next time
#split_idx = dataset.get_idx_split()
#train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
# graph = dataset[0]
# graph_coo = graph.edge_index
# print(graph_coo.shape)

graph = dataset[0]
print(graph) ## to see the pyg papers-100m data' format
graph_coo = graph.edge_index
print(graph_coo.shape)  

## convert the dircted graph to undirected graph
from torch_geometric.utils import to_undirected
un_coo = to_undirected(graph_coo)
print(un_coo.shape)

### get the node and edge numbers
nodes = graph.y.numpy()
# import numpy as np
# coo = graph_coo.numpy()
print(nodes.shape)
# print(coo.shape)
node_nums = nodes.shape[0]
print(node_nums)
edge_nums = un_coo.shape[1]
print(edge_nums)


# sort the un_direct graph  maybe it is not neccessary but is great for count the csr format
import numpy as np
coo = un_coo.numpy()
coo_sort = coo.T[np.lexsort(coo[0,None])].T
print(coo_sort.shape)
# numpy format had done
 
# write the COO TXT
with open('/hdd1/datasets_gujy/ogb_datasets/COO_sorted_file.txt','w') as fp:
    fp.write("nodes_nums:"+str(node_nums)+'--'+ " "+str(edge_nums)+'\n')
    for i in range(edge_nums):
        fp.write(str(coo_sort[0,i])+' '+str(coo_sort[1,i])+' '+str(1)+'\n')
    fp.close()
print("Sorted had done!")


# count the csr row offsets
row_index= coo_sort[0,:]
col_index= coo_sort[1,:]
row_offsets = np.zeros([1,node_nums+1]).reshape(-1,)
for i in range(edge_nums):
    row_offsets[row_index[i]+1] += 1
print(row_offsets.shape)
#print(row_offsets)
for i in range(1,row_offsets.shape[0]):
    row_offsets[i] = row_offsets[i]+row_offsets[i-1]
#print(row_offsets)
# convert to the int type
row_offsets = row_offsets.astype(int)
# save CSR txt
col = col_index.tolist()
rf = row_offsets.tolist()
#len(col)
with open(save_path+'row_offsets.txt','w') as fp:
    fp.write("node_nums:"+str(node_nums)+';'+ " edge_nums: "+str(edge_nums)+" the value is 1 to present connect; attention: the length of row offsets is node_length + 1"+'\n')
    [fp.write(str(item)+'\n') for  item in rf]
    fp.close()
with open(save_path+'/colum_index.txt','w') as fp:
    fp.write("node_nums:"+str(node_nums)+'--'+ " edge_nums: "+str(edge_nums)+" the value is 1 to present connect "+'\n')
    [fp.write(str(item)+' '+str(1)+'\n') for item in col]
    fp.close()
