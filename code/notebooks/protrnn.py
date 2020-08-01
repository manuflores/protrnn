import torch
import torch.nn as nn 
import torch.functional as F 

import pandas as pd 


def read_data_pfam(fname, select_cols = None):
    """
    Helper function to load protein sequences. 
    """
    f = open(fname).read().strip().split("\n")

    col_names = f[0].split(",")

    # Substract one for the header
    n_seqs = len(f) - 1

    # Loop through sequences adding each one as a column to the df
    annot_seqs = [f[i].split(",") for i in range(1, n_seqs)]

    # Initialize pandas dataframe with sequences
    df = pd.DataFrame(annot_seqs, columns=col_names)

    if select_cols is not None:
        df = df[select_cols]

    return df


def get_random_training_protein(data, categories, category_weights,
                                category_col_name = 'family_id'): 
    """
    Returns a random protein for training. This function is designed to help 
    for RNN classification task. 
    """
    
    # Choose category 
    category = np.random.choice(categories, size = 1, p = category_weights) 
    
    #print(category)
    # Get index from category and turn into torch.Variable
    category_tensor = Variable(torch.LongTensor([categories.index(category)]))
    
    # Pick random protein from given category 
    df_cat = data[data[category_col_name] == category[0]]
    
    # Get protein
    sample_prot = df_cat.sample()
    prot = sample_prot['sequence'].values[0]
    prot_name = sample_prot['sequence_name'].values[0]
    
    # Turn sequence to list of indices 
    prot_indices = Variable(torch.LongTensor(aminoacid_encoder(prot)))
    
    return category_tensor, prot_indices, prot_name



def category_from_output(output):
    
    """
    Given an output vector (in log-softmax format), 
    returns the category with the highest probability 
    and its corresponding index. 
    """
    top_value, top_ix = output.data.topk(1) 
    
    # Extract the value from array
    category_ix = top_ix[0][0]
    
    return all_categories[category_ix], category_ix


def initialize_network_weights(net, method = 'kaiming'): 
    """
    Initialize fully connected and convolutional layers' weights
    using the Kaiming (He) or Xavier method. 
    This method is recommended for ReLU / SELU based activations.
    """

    torch.manual_seed(4)

    if method == 'kaiming': 
        for module in net.modules():

            if isinstance(module, nn.Linear): 
                nn.init.kaiming_uniform_(module.weight)
                nn.init.uniform_(module.bias)

    elif method == 'xavier':
        for module in net.modules():

            if isinstance(module, nn.Linear): 
                nn.init.kaiming_uniform_(module.weight)
                nn.init.uniform_(module.bias)

    else: 
        print('Method not found. Only valid for kaiming or xavier initialization.')

    return net



class RNN(nn.Module): 
    
    def __init__(self, input_size, hidden_size, output_size, n_layers = 1): 

        super(RNN, self).__init__()
        
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_layers = n_layers 
        
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # 
        self.gru = nn.GRU(
            input_size = hidden_size,
            hidden_size= hidden_size,
            num_layers = n_layers
        )
        
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def init_hidden(self): 
    	
        # Use batch_size = 1
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        

# class ProteinDataset(Dataset): 
#     """
#     Assumes the input dataframe has an ordered set of indices. 
#     If it is not, call df.reset_index() before using function. 
#     """
    
#     def __init__(self, df, transform = False, supervised = False, hot_encode = False): 
        
#         self.df = df
#         self.transform = transform 
#         self.supervised = supervised 
#         self.hot_encode = hot_encode
        
#         aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
#         # Leave index 0 to encode a non-canonical aminoacid
#         self.aa_to_ix = dict(zip(aminoacids, np.arange(1, 21)))
#         self.ix_to_aa = dict(zip(np.arange(1, 21), aminoacids))
    
#     def aminoacid_to_ix(self, seq): 
#         "Returns a list of indices from a string of aminoacids."
#         return [self.aa_to_ix.get(aa, 0) for aa in seq]
    
#     def ix_to_aminoacid(ixs): 
#         "Returns a list of aminoacids from a list of indices."
#         return [self.ix_to_aa.get(ix, 0) for ix in ixs]
    
#     def __len__(self): 
#         return len(self.df)
    
#     def __getitem__(self, ix): 
        
#         if type(ix) == torch.Tensor: 
#             ix = ix.tolist()
            
#         # Get protein sequence 
#         prot = self.df.loc[ix, 'sequence']
        
#         # Label encode protein, use CharTensor for 8bit integers
#         prot_encoded = torch.tensor(self.aminoacid_to_ix(prot), dtype=torch.long)
        
#         if self.hot_encode: 
            
        
#         if self.supervised: 
#             label = self.df.loc[ix, 'family_id']
            
#             return prot_encoded, label 
        
#         return prot_encoded         



class RNN_GRU_concise(nn.Module): 
    
    def __init__(self, input_size, embedding_size, hidden_size,
                 output_size, n_layers = 1):
        
        super(RNN, self).__init__()
        
        self.input_size = input_size 
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_layers = n_layers 
        self.log_softmax = nn.LogSoftmax(dim= 1)
        
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # 
        self.gru = nn.GRU(
            input_size = embedding_size,
            hidden_size= hidden_size,
            num_layers = n_layers
        )
    
        self.decoder = nn.Linear(hidden_size, output_size)
      
    # torch initializes with uniform distribution by default.
        
    # def init_hidden(self): 
    #     # Use batch_size = 1
    #     return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    
    def aa_encoder(self, input): 
        projected = self.embedding(input)
        
        return projected 
    

    def forward(self, input, hidden): 
        
        embedding_tensor = self.embedding(input)

        embedding_tensor = embedding_matrix.view(len(input), 1, self.embedding_size)

        
        # Input shape(seq_len = len(sequence), batch_size = 1, input_size = -1)

        sequence_of_hiddens, last_hidden = self.gru(embedding_tensor)
        
        output_rnn = sequence_of_hiddens.view(len(input), self.hidden_size)

        output = self.decoder(output_rnn)
        
        # LogSoftmax the output for numerical stability
        output = self.log_softmax(output)
        return output, hidden
        