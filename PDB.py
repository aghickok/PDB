# -*- coding: utf-8 -*-
import utils
import numpy as np
import matplotlib.pyplot as plt
import queue
from queue import PriorityQueue
import networkx as nx
import matplotlib
import math
from gudhi.wasserstein import wasserstein_distance as W

##############################################
# RU decomposition class (and helper classes)
##############################################

                    
class sparseMatrix:
    '''
    Represents sparse binary matrices.
    '''
    def __init__(self, n, col = True, print_progress = False):
        # Initialize empty (n x n) sparse matrix. Use col = True if you want to add
        # columns quickly, or col = False if you want to add rows quickly.
        self.reduced = False
        self.print_progress = print_progress
        self.n = n
        
        # If col = False, will store the transpose of A.
        self.col = col
        
        # col_array[i] is the index in col_list of the linked list that represents column i in the current matrix
        #self.col_array = [j for j in range(n)]
        
        # row_array[i] represents the original ith row. It stores its current 
        # index in the matrix and also stores the reverse link, i.e. row_array[i] is
        # a tuple s.t. row_array[i][0] = current index of the original ith row,
        # and row_array[i][1] = original index of what is now the ith row
        self.row_array = [[i, i] for i in range(n)]

        # Initialize empty linked list for each column.
        # col_list[i] stores what was originally column i. The nodes in the list
        # have values equal to the ORIGINAL row indices with non-zero entries.
        # If col = False, should just store the transpose and remember later when adding rows/cols, etc
        self.col_list = [utils.linkedList() for i in range(n)]

    def __eq__(self, other):
        # Returns True if self and other represent the same underlying matrix
        assert self.col and other.col, "haven't implemented for col = False yet"
        if self.n != other.n:
            return False
        for j in range(self.n):
            self_entries = self.get_col_nonzero_entries(j)
            other_entries = other.get_col_nonzero_entries(j)
            if not set(self_entries) == set(other_entries):
                print("col ", j, " is incorrect")
                print(self_entries)
                print(other_entries)
                return False
        return True        
    
    def __getitem__(self, idx):
        # Returns A_{ij} where i = idx[0] and j = idx[1]
        i = idx[0]
        j = idx[1]
        if self.col:
            col_j_list = self.get_col(j)
            orig_i_idx = self.row_array[i][1]
            if col_j_list.has(orig_i_idx):
                return 1
            else:
                return 0
        else:
            row_i_list = self.get_row(i)
            orig_j_idx = self.row_array[j][1]
            if row_i_list.has(orig_j_idx):
                return 1
            else:
                return 0
                
    def __mul__(self, other):
        assert other.n == self.n, "matrix dimensions have to agree"
        assert self.col and not other.col, "haven't implemented when self.col = False or when other.col = True"
        prod = sparseMatrix(self.n)
        for k in range(self.n):
            col = self.get_col(k)
            row = other.get_row(k)
            col_curr = col.head
            while(col_curr is not None):
                i = self.get_row_idx(col_curr)
                row_curr = row.head
                while(row_curr is not None):
                    j = other.get_col_idx(row_curr)
                    prod.plus_1(i, j)
                    row_curr = row_curr.next
                col_curr = col_curr.next
        return prod
                    
    def add_col(self, i, j):
        # Add what is currently column i to what is currently column j.
        assert self.col, "Can't add columns if col = False"
        
        # Update inverse_low dictionary
        if self.reduced:
            low_j = self.low(j)
            if low_j != -1:
                if low_j != -1 and len(self.inverse_low[low_j]) == 1:
                    del self.inverse_low[low_j]
                else:
                    self.inverse_low[low_j].remove(j)
                    
            col_j = self.get_col(j)
            col_j += self.get_col(i)
            
            # Finish updating inverse_low
            new_low_j = self.low(j)
            if new_low_j != -1:
                if new_low_j in self.inverse_low:
                    self.inverse_low[new_low_j].append(j)
                else:
                    self.inverse_low.update({new_low_j : [j]})
        else:
            col_j = self.get_col(j)
            col_j += self.get_col(i)
            
    def add_row(self, i, j):
        # Add row i to row j. Can only be called if col = False. In that case
        # we're actually storing the transpose, so add column i to column j
        # as in add_col
        assert not self.col, "Haven't implemented add_row if col = True"
        row_j = self.get_row(j)
        row_j += self.get_row(i)
    
    def copy(self):
        # Returns a sparseMatrix that represents the same underlying matrix. Deep copy
        copy = sparseMatrix(self.n, col = self.col)
        #copy.col_array = self.col_array.copy()
        copy.row_array = [pair.copy() for pair in self.row_array]
        for i in range(self.n):
            # Copy what's currently ith column (or row) in A to copy.col_list[i]
            self_curr = self.col_list[i].head
            if self_curr is not None:
                copy.col_list[i].head = utils.Node(self_curr.data)
                copycurr = copy.col_list[i].head
                while self_curr.next is not None:
                    copycurr.next = utils.Node(self_curr.next.data)
                    self_curr = self_curr.next
                    copycurr = copycurr.next
        return copy
        
    def get_col(self, i):
        # Gets the current ith col, stored as linked list
        assert self.col, "Haven't implemented get_col for col = False"
        #return self.col_list[self.col_array[i]]
        return self.col_list[i]
    
    def get_col_idx(self, node):
        # Get current col idx of a node. Only call if self.col = False
        assert not self.col, "Haven't implemented get_col_idx for col = True"
        if node is None:
            return None
        else:
            return self.row_array[node.data][0]
    
    def get_col_nonzero_entries(self, i):
        # Get the (current) row indices of the nonzero entries in col i. Return list
        if self.col:
            row_indices = []
            curr = self.get_col(i).head
            while curr is not None:
                row_indices.append(self.get_row_idx(curr))
                curr = curr.next
            return row_indices
        else:
            # This is slow if col = False
            row_indices = []
            # Check each row to see if there's a node in col i
            for j in range(self.n):
                curr = self.get_row(j).head
                while curr is not None:
                    if self.get_col_idx(curr) == i:
                        row_indices.append(j)
                        break
                    curr = curr.next
            return row_indices
                          
    def get_row_nonzero_entries(self, i):
        if self.col:
            # Slow if col = True.
            col_indices = []
            for j in range(self.n):
                curr = self.get_col(j).head
                while curr is not None:
                    if self.get_row_idx(curr) == i:
                        col_indices.append(j)
                        break
                    curr = curr.next
            return col_indices
        else:
            col_indices = []
            curr = self.get_row(i).head
            while curr is not None:
                col_indices.append(self.get_col_idx(curr))
                curr = curr.next
            return col_indices
        
    def get_row(self, i):
        # Gets the current ith row, stored as linked list
        assert not self.col, "Have't implemented for col = True"
        #return self.col_list[self.col_array[i]]
        return self.col_list[i]
    
    def get_row_idx(self, node):
        # Get current row index of a node.
        assert self.col, "Haven't implemented for self.col = False"
        if node is None:
            return None
        else:
            return self.row_array[node.data][0]

    def identity(n, col= True):
        # Returns sparseMatrix representation of the nxn identity matrix
        I = sparseMatrix(n, col = col) # zero matrix
        for j in range(n):
            I.col_list[j].append(utils.Node(j))
        return I

    def is_positive(self, j):
        # Returns True if column j is a zero column (correspondingly, simplex j
        # is positive), false otherwise. Only works when col = True because in 
        # the vineyard context, we only ever apply this for R.
        #return self.col_list[self.col_array[j]].head is None
        return self.get_col(j).head is None
    
    def is_upper(self):
        if self.col:
            for i in range(self.n):
                col = self.get_col(i)
                curr = col.head
                while(curr is not None):
                    if self.get_row_idx(curr) > i:
                        return False
                    curr = curr.next
            return True
        else:
            for i in range(self.n):
                row = self.get_row(i)
                curr = row.head
                while(curr is not None):
                    if self.get_col_idx(curr) < i:
                        return False
                    curr = curr.next
            return True
    
    def plus_1(self, i, j):
        # Only ever gets called when multiplying matrices, not when the matrix is reduced.
        # Does NOT preserve the self.inverse_low mapping
        assert self.col, "Haven't implemented for self.col = False"
        #old_entry = self[i, j]
        curr = self.get_col(j).head
        orig_i = self.row_array[i][1]
        if curr is None:
            self.get_col(j).head = utils.Node(orig_i)
        elif curr.data == orig_i:
            self.get_col(j).head = curr.next
        elif curr.data > orig_i:
            new_head = utils.Node(orig_i)
            new_head.next = curr
            self.get_col(j).head = new_head
        else:
            while(curr.next is not None and curr.next.data < orig_i):
                curr = curr.next
            # Now curr is the last node s.t. curr.data < orig_i
            if curr.next is not None and curr.next.data == orig_i:
                # Then remove curr.next from list
                curr.next = curr.next.next
            else:
                new_node = utils.Node(orig_i)
                new_node.next = curr.next
                curr.next = new_node

    def swap_cols(self, i, j):
        # Exchange cols i and j. Implementation depends on if col = True.
        if self.col:
            if self.reduced:
                low_i = self.low(i)
                low_j = self.low(j)
                if low_i != -1:
                    self.inverse_low[low_i].remove(i)
                    self.inverse_low[low_i].append(j)
                if low_j != -1:
                    self.inverse_low[low_j].remove(j)
                    self.inverse_low[low_j].append(i)
                    
                #i_idx = self.col_array[i]
                #self.col_array[i] = self.col_array[j]
                #self.col_array[j] = i_idx
            #else:  
                # i_idx = self.col_array[i]
                # self.col_array[i] = self.col_array[j]
                # self.col_array[j] = i_idx
            self.col_list[i], self.col_list[j] = self.col_list[j], self.col_list[i]
            
        else:
            # exhange the "rows" i and j as in swap_rows(i, j), because 
            # we're actually storing the transpose
            i_orig_idx = self.row_array[i][1]
            j_orig_idx = self.row_array[j][1]
            self.row_array[i_orig_idx][0] = j # current index of j_orig_idx
            self.row_array[j_orig_idx][0] = i # current index of i_orig_idx
            self.row_array[i][1] = j_orig_idx
            self.row_array[j][1] = i_orig_idx
    
    def swap_rows(self, i, j):
        # Exchange what are currently rows i and j. Implementation depends on if col = True. 
        if self.col:
            if self.reduced:
                k_list = []
                if i in self.inverse_low:
                    k_list = self.inverse_low[i]
                if (i+1) in self.inverse_low:
                    k_list = k_list + self.inverse_low[i+1]
                k_list = set(k_list)    # set of cols k s.t. low(k) = i or i+1
                old_lows = {k : self.low(k) for k in k_list}    # every self.low(k) is either i or i+1
                
                i_orig_idx = self.row_array[i][1]
                j_orig_idx = self.row_array[j][1]
                self.row_array[i_orig_idx][0] = j # current index of j_orig_idx
                self.row_array[j_orig_idx][0] = i # current index of i_orig_idx
                self.row_array[i][1] = j_orig_idx
                self.row_array[j][1] = i_orig_idx
                
                # Update inverse_low after swap
                for k in k_list:
                    new_low_k = self.low(k) # either i or i+1
                    if new_low_k != old_lows[k]:
                        if len(self.inverse_low[old_lows[k]]) == 1:
                            del self.inverse_low[old_lows[k]]
                        else:
                            self.inverse_low[old_lows[k]].remove(k)
                        if new_low_k in self.inverse_low:
                            self.inverse_low[new_low_k].append(k)
                        else:
                            self.inverse_low.update({new_low_k : [k]})
            else:
                i_orig_idx = self.row_array[i][1]
                j_orig_idx = self.row_array[j][1]
                self.row_array[i_orig_idx][0] = j # current index of j_orig_idx
                self.row_array[j_orig_idx][0] = i # current index of i_orig_idx
                self.row_array[i][1] = j_orig_idx
                self.row_array[j][1] = i_orig_idx
        else:
            # exchange the "columns" i and j as in swap_cols, because
            # we're actually storing the transpose
            # i_idx = self.col_array[i]
            # self.col_array[i] = self.col_array[j]
            # self.col_array[j] = i_idx
            self.col_list[i], self.col_list[j] = self.col_list[j], self.col_list[i]
    
    def low(self, j):
        # Returns current row index of lowest 1 in what is currently column j.
        # In vineyard context, only gets called for R, so this code only works
        # if col = True. Returns -1 if there are no 1s in col j. (MAY WANT TO CHANGE THIS TO NONE)
        if self.col:
            #col_j_list = self.col_list[self.col_array[j]]
            col_j_list = self.get_col(j)
            curr_node = col_j_list.head
            curr_max = -1
            while curr_node is not None:
                curr_row = self.row_array[curr_node.data][0]
                if curr_row > curr_max:
                    curr_max = curr_row
                curr_node = curr_node.next
            return curr_max

    def print(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self[i, j], end = "\t")
            print("\n")
            
    def reduce(self):
        '''
        Only ever gets called for D, so only needs to work when col = True.
        Implements simplex pairing reduction algorithm from paper. D gets reduced to R.
        Returns the matrix U s.t. if D is the original matrix and R is the reduced matrix,
        then D = RU where U is upper-triangular.
        When computing U, I'm making the assumption that D and R are square (which is true in the vineyard context)
        '''
        assert self.col, "Only implemented for self.col = True"
        if self.print_progress: print("started reducing")
        U = sparseMatrix.identity(self.n, col = False)
        self.inverse_low = {}
        for j in range(self.n):
            if self.print_progress: print(f"col {j}")
            i = self.low(j)
            while i in self.inverse_low:
                j_ = self.inverse_low[i][0]
                #if self.print_progress: print(f"adding col {j_} to {j}")
                self.add_col(j_, j)
                U.add_row(j, j_)
                i = self.low(j)
            if i != -1:
                self.inverse_low.update({i : [j]})
        self.reduced = True
        return U
            
    def set_zero(self, i, j):
        '''
        Set A_{ij} equal to 0, if it isn't already. To do this: If col = True, 
        remove the node with value row_array[i][1] (if it's there) from the linked list 
        representing what is currently column j. If col = False, remove the node
        with value row_array[j][1] (if it's there) from the linked list
        representing what is currently row i.
        '''
        if self.col:
            #col_j_list = self.col_list[self.col_array[j]]
            col_j_list = self.get_col(j)
            col_j_list.remove(self.row_array[i][1])
        else:
            #row_i_list = self.col_list[self.col_array[i]]
            row_i_list = self.get_row(i)
            row_i_list.remove(self.row_array[j][1])
            
class RU:
    def __init__(self, simplices, debug_mode = False, print_progress = False):
        '''
        simplices: list of Simplex objects.
        '''
        self.print_progress = print_progress
        self.debug_mode = debug_mode
            
        D = RU.boundary_matrix(simplices)
        
        if debug_mode: self.D = D.copy()   # Need to copy D before it gets reduced
        
        self.U = D.reduce() # The reduce function reduces D in place to R and returns the upper triangular matrix U s.t. D = R*U
        self.R = D
        self.n = len(simplices)
        
        if debug_mode: 
            self.check_decomposition()
            print("initial reduce is correct")
    
    def boundary_matrix(simplices):
        '''
        simplices: List of Simplex objects
        
        Returns: D (sparseMatrix). The boundary matrix for the simplices ordered as they are in simplices, over field F_2.
        '''
        C = {}
        for i, spx in enumerate(simplices):
            C[spx.nodes] = i
        #print(C)
        
        n = len(simplices)
        bdry_matrix = sparseMatrix(n)
        for i, spx in enumerate(simplices):
            nodes = spx.nodes
            #print("\ncomputing boundary of simplex with nodes: ", nodes)
            dim = len(nodes) - 1
            if not dim == 0:
                bdry = bdry_matrix.get_col(i)
                for j in range(dim + 1):
                    face = tuple([nodes[k] for k in range(dim+1) if not k == j])
                    #print("face: ", face)
                    bdry.insert(utils.Node(C[face]))
        return bdry_matrix
        
    def check_decomposition(self, i = None, case = None):
        # only used for testing
        if i is not None and case is not None:
            additional_info = f" after swapping {i} and {i+1} (case {case})"
        elif i is not None:
            additional_info = f" after swapping {i} and {i+1}"
        elif case is not None:
            additional_info = f" after a case {case} swap"
        else:
            additional_info = ""
        assert self.D.is_upper(), "D isn't upper triangular anymore (not a filtration)" + additional_info
        assert self.U.is_upper(), "U isn't upper triangular anymore" + additional_info
        assert self.is_R_reduced(), "R isn't reduced anymore" + additional_info
        assert self.D == self.R*self.U, "RU != D" + additional_info
    
    def get_spx_pairs(self):
        pairs = {}
        for j in range(self.n):
            if self.R.is_positive(j):
                pairs.update({j: None})
            else:
                i = self.R.low(j)
                pairs.update({i : j})
        return pairs
     
    def is_R_reduced(self):
        # only used for testing
        pairs = {}
        for j in range(self.n):
            i = self.R.low(j)
            if i != -1:
                if i in pairs:
                    return False
                else:
                    pairs.update({i : j})
        return True
            
    #def swap(self, i, dim_i, dim_iplus):
    def swap(self, spx_a, spx_b):
        '''
        If P is the permutation matrix that swaps i, i+1, then PDP is the matrix
        obtained by swapping rows i, i+1 and cols i, i+1 of D. Update R, U
        to be a decomposition for PDP.
        spx_a, spx_b: Simplex objects that are getting swapped.
        
        #dim_i = dimension of current ith simplex
        #dim_iplus = dimension of current (i+1)st simplex
        
        Returns True if we need to update the simplex pairings in the vineyard, False otherwise.
        '''
        if spx_a.curr_idx < spx_b.curr_idx:
            i = spx_a.curr_idx
            dim_i = spx_a.dim
            dim_iplus = spx_b.dim
        else:
            i = spx_b.curr_idx
            dim_i = spx_b.dim
            dim_iplus = spx_a.dim
        
        need_update = False
        
        if self.debug_mode:
            self.D.swap_cols(i, i+1)
            self.D.swap_rows(i, i+1)
            
        if dim_i != dim_iplus:
            case = "0"
            self.R.swap_cols(i, i+1)
            self.R.swap_rows(i, i+1)
            self.U.swap_cols(i, i+1)
            self.U.swap_rows(i, i+1)
        else:    
            i_pos = self.R.is_positive(i)
            iplus_pos = self.R.is_positive(i+1)
            
            if i_pos and iplus_pos:
                # Case 1: i and i+1 are positive simplices
                # In this case, can set U_{i, i+1} = 0 and still have an RU decomp.
                
                self.U.set_zero(i, i+1)
                
                # Now PUP will be upper triangular.
                self.U.swap_rows(i, i+1)
                self.U.swap_cols(i, i+1)
                
                # Figure out if we're in Case 1.1: There are cols k, l s.t. 
                # low_R(k) = i and low_R(l) = i+1 and R[i, l] = 1.
                case11 = False
                if (i+1) in self.R.inverse_low:
                    l = self.R.inverse_low[i+1][0]
                    if self.R[i, l] == 1 and i in self.R.inverse_low:
                        k = self.R.inverse_low[i][0]
                        case11 = True
                        
                if case11:
                    # Case 1.1. Now compute PRP
                    self.R.swap_rows(i, i+1)
                    self.R.swap_cols(i, i+1)
                    
                    # Reduce PRP.
                    if k < l:
                        # Case 1.1.1
                        case = "1.1.1"
                        self.R.add_col(k, l)    # add col k to col l in PRP
                        self.U.add_row(l, k)    # add row l to row k in PUP
                    if l < k:
                        # Case 1.1.2
                        case = "1.1.2"
                        need_update = True
                        self.R.add_col(l, k)
                        self.U.add_row(k, l)
                else:
                    # Case 1.2. Then (PRP)(PUP) is an RU decomposition for PDP.
                    case = "1.2"
                    if i not in self.R.inverse_low: need_update = True
                    self.R.swap_rows(i, i+1)
                    self.R.swap_cols(i, i+1)
                    
            elif not i_pos and not iplus_pos:
                # Case 2: i and i+1 are both negative simplices. In this case, rows
                # i and i+1 can't contain the lowest 1s of an columns, so PRP is 
                # reduced. Just need to fix PUP.
                if self.U[i, i+1] == 1:
                    # Case 2.1
                    if self.R.low(i) < self.R.low(i+1):
                        # Case 2.1.1
                        case = "2.1.1"
                        self.R.add_col(i, i+1)
                        self.R.swap_rows(i, i+1)
                        self.R.swap_cols(i, i+1)
                        
                        self.U.add_row(i+1, i)
                        self.U.swap_rows(i, i+1)
                        self.U.swap_cols(i, i+1)
                    else:
                        # Case 2.1.2: self.R.low(i) > self.R.low(i+1)
                        case = "2.1.2"
                        need_update = True
                        self.R.add_col(i, i+1)
                        self.R.swap_rows(i, i+1)
                        self.R.swap_cols(i, i+1)
                        self.R.add_col(i, i+1)
                        
                        self.U.add_row(i+1, i)
                        self.U.swap_rows(i, i+1)
                        self.U.swap_cols(i, i+1)
                        self.U.add_row(i+1, i)
                else:
                    # Case 2.2
                    case = "2.2"
                    self.R.swap_cols(i, i+1)
                    self.R.swap_rows(i, i+1)
                    
                    self.U.swap_cols(i, i+1)
                    self.U.swap_rows(i, i+1)
                    
            elif not i_pos and iplus_pos:
                # Case 3: i is a negative simplex and i+1 is a postive simplex
                if self.U[i, i+1] == 1:
                    # Case 3.1
                    case = "3.1"
                    need_update = True
                    self.R.add_col(i, i+1)
                    self.R.swap_rows(i, i+1)
                    self.R.swap_cols(i, i+1)
                    self.R.add_col(i, i+1)
                    
                    self.U.add_row(i+1, i)
                    self.U.swap_rows(i, i+1)
                    self.U.swap_cols(i, i+1)
                    self.U.add_row(i+1, i)
                else:
                    # Case 3.2
                    case = "3.2"
                    self.R.swap_cols(i, i+1)
                    self.R.swap_rows(i, i+1)
                    
                    self.U.swap_cols(i, i+1)
                    self.U.swap_rows(i, i+1)
            else:
                # Case 4: i is a positive simplex and i+1 is a negative simplex
                case = "4"
                self.U.set_zero(i, i+1)
                self.R.swap_cols(i, i+1)
                self.R.swap_rows(i, i+1)
                
                self.U.swap_cols(i, i+1)
                self.U.swap_rows(i, i+1)
        
        if self.debug_mode: self.check_decomposition(i, case)
        return need_update
            
############################################################
# Simplex object
############################################################
class Simplex:
    # Represents a simplex with a 2-dim array of filtration values
    def __init__(self, nodes, val):
        # nodes: list of nodes
        # val: 2d array of filtration values
        assert val.ndim == 2
        nodes.sort()
        self.nodes = tuple(nodes)
        self.val = val
        self.curr_idx = None
        self.dim = len(nodes) - 1
        self.interpolator = None
    
    def __call__(self, x, y):
        if self.interpolator is not None:
            return self.interpolator(x, y)
        else:
            print("No interpolator")
            return None
        
    def __lt__(self, other):
        # Only gets called for sorting sclimplices at the initial vertex of the triangulation
        # Assume that all simplices have different filtration values at the initial vertex.
        if self.val[(0, 0)] == other.val[(0, 0)]:
            return self.dim < other.dim
        return self.val[(0, 0)] < other.val[(0, 0)]
    
    def coeffs(self, u, v, w):
        # only for testing
        ''' 
        u, v, w: tuples. coordinates of the vertices of some triangle T, 
        where u is the SW vertex.
        Assumes that the positions of the vertices are the same as their indices.
        Returns coefficients a, b, d s.t. the filtration value at (x, y) in T
        is z = d + a*x + b*y
        '''
        A = np.array([[u[0], u[1], 1], [v[0], v[1], 1], [w[0], w[1], 1]])
        b = np.array([self.val[u], self.val[v], self.val[w]])
        x = np.linalg.solve(A, b)
        a = x[0]
        b = x[1]
        d = x[2]
        return a, b, d

    def intersection(self, other_spx, t, idx1, idx2):
        '''
        other_spx: Simplex object
        t: float in [0, 1]. We think of [0, 1] as parameterizing the edge in
        the triangulation from vertex at idx1 to the vertex at idx2.
        idx1: tuple. index in self.val array
        idx2: tuple. index in sel.val array that's adjacent to idx1
        
        Returns the coordinates (t_intersect, y) of the intersection, if 
        there's an intersection in [t, 1). Else return None.
        If the filtration values are inputted as Fraction objects, then this returns exact intersection (in Fraction coordinates).
        '''
        if (other_spx.val[idx1] - self.val[idx1])*(other_spx.val[idx2] - self.val[idx2]) < 0:
            # The simplices swap in the interval [idx1, idx2] iff their orders at the endpoints are different
            denom = other_spx.val[idx2] - other_spx.val[idx1] + self.val[idx1] - self.val[idx2]
            assert denom != 0
            t_intersect = (self.val[idx1] - other_spx.val[idx1])/denom
            if t_intersect >= t and t_intersect < 1:
                y_intersect = (1 - t_intersect)*self.val[idx1]  + (t_intersect)*self.val[idx2]
                return [t_intersect, y_intersect]
            else:
                return None
        elif other_spx.val[idx1] == self.val[idx1]:
            # Equal at t = 0
            return [0, self.val[idx1]]
        else:
            return None
    
    def T_intersection(self, other_spx, u, v, w):
        # only for testing
        a1, b1, d1 = self.coeffs(u, v, w)
        a2, b2, d2 = other_spx.coeffs(u, v, w)
        if (w[1] - u[1]) + (v[1] - u[1]) == 2:
            # then T is an upper triangle (above the diagonal)
            xv = u[0]
            yh = u[1] + 1
        else:
            xv = u[0] + 1
            yh = u[1]
        
        intersections =  []
        
        # Intersection with vertical edge?
        if b1 != b2:
            y = (d2 + (a2 - a1)*xv - d1)/(b1 - b2)
            if u[1] <= y <= u[1] + 1:
                intersections.append((xv, y))
        
        # Intersection with horizontal edge?
        if a1 != a2:
            x = (d2 + (b2 - b1)*yh - d1)/(a1 - a2)
            if u[0] <= x <= u[0] + 1:
                intersections.append((x, yh))
                
        # intersection with diagonal edge?
        if (a1 + b1 - a2 - b2) != 0:
            x = (d2 - d1 + (b2 - b1)*(u[1] - u[0]))/(a1 + b1 - a2 - b2)
            y = (x - u[0]) + u[1]
            if u[0] <= x <= u[0] + 1:
                intersections.append((x, y))
        return intersections
            
        
###############################################################################
# Helper DCEL classes and functions
###############################################################################
class Vertex:
    def __init__(self, pos, idx = None):
        # inc_edges are half edges whose origin is the vertex
        self.pos = pos
        self.inc_edges = []
        self.idx = idx
        
    def __str__(self):
        # Only for testing
        edge_strs = ""
        for e in self.inc_edges:
            edge_strs += (e.pos() + ", ")
        return f"Position: {self.pos}, Incident edges: " + edge_strs
    
    def angle(self, v, e):
        # v: Vertex object.
        # e: an edge whose origin is self
        # Returns: alpha, the angle by which to rotate counterclockwise to get from vector (self, w) to edge e
        assert e.org == self
        ux, uy = self.pos
        vx, vy = v.pos
        theta_v = np.angle(complex(vx - ux, vy - uy)) % (2*math.pi)
        wx, wy = e.dest().pos
        theta_e = np.angle(complex(wx - ux, wy - uy)) % (2*math.pi)
        alpha = (theta_v - theta_e) % (2*math.pi)
        return alpha
        
    def get_inc_edge(self, T, v = None):
        # T: triangle in triangulation.
        # v: Vertex object.
        # Returns the edge, if it exists, that has its origin at self
        # and is incident to triangle T. In "generic" case where line segments don't share endpoints, the edge is unique. Otherwise, choose edge that's closest in (signed) angle to the vector (self, v)
        inc_T_edges = []
        for e in self.inc_edges:
            #if e.T == T: return e
            if e.T == T: 
                inc_T_edges.append(e)
        num_edges = len(inc_T_edges)
        if num_edges == 0:
            return None
        elif num_edges == 1:
            return inc_T_edges[0]
        else:
            assert v is not None, "incident edge to self and triangle isn't unique, need comparison vertex"
            angles = [self.angle(v, e) for e in inc_T_edges]
            i = np.argmin(angles)
            return inc_T_edges[i]
    
    def get_all_inc_faces(self):
        faces = {e.left for e in self.inc_edges if e.left is not None}
        return faces
            
    def get_inc_face(self):
        # Returns an arbitrary face that is incident to the vertex, if such a face exists (return the first one we find)
        f = None
        if len(self.inc_edges) > 0:
            f = None
            for e in self.inc_edges:
                f = e.left
                if f is not None: break
        return f
     
class Edge:
    # directed edge
    def __init__(self, org):
        self.org = org # origin vertex
        self.twin = None # half edge on the other side
        self.left = None # face to its left
        self.next = None # next edge in face
        self.prev = None # previous edge in face
        self.T = None # (integer) index of triangle in triangulation that it's incident to, if any
        self.spx_pair = None # list of tuples of Simplex objects that are swapping along this edge
        
    def __str__(self):
        # Only for testing
        str_ = self.pos()
        if self.prev is not None:
            str_ += (", Previous: " + self.prev.pos())
        if self.next is not None:
            str_ += (", Next: " + self.next.pos())
        if self.left is not None:
            str_ += f", Bounds face incident to {self.left.inc_edge.pos()}"
        return str_
    
    def dest(self):
        return self.twin.org
    
    def pos(self):
        # Only for testing, used by various __str__ methods
        return "(" + str(self.org.pos) + ", " + str(self.twin.org.pos) + ")"
    
class Face:
    def __init__(self, inc_edge):
        self.inc_edge = inc_edge
        self.spx_pairs = None
    
    def __str__(self):
        # Only for testing
        str_ = ""
        edges = self.edges()
        for e in edges:
            str_ += (e.pos() + ", ")
        return str_
            
    def edges(self):
        e0 = self.inc_edge
        edges = [e0]         
        e = e0.next
        while e != e0:
            edges.append(e)
            e = e.next
        return edges
    
    def vertices(self):
        vertices = [e.org for e in self.edges()]
        return vertices

    
##############################################################################
# Crossing class for Bentley-Ottman planesweep algorithm
##############################################################################
class Crossing:
    def __init__(self, spx1, spx2, t, y):
        self.spx1 = spx1
        self.spx2 = spx2
        self.t = t
        self.y = y
        
    def __lt__(self, other_event):
        # Sort by time first.
        if self.t < other_event.t:
            return True
        elif self.t > other_event.t:
            return False
        else:
            # In this case, events occur at the same time.
            # Prioritize by index of spx1 (lower first, but it doesn't really matter)
            return self.spx1.curr_idx < other_event.spx1.curr_idx
    
    def __str__(self):
        return f"Time: {self.t} \ny-coordinate: {self.y}"

###############################################################################
# DCEL class for storing (and constructing) the line arrangement
###############################################################################

class DCEL: 
    def __init__(self, simplices = None):
        self.edges = []
        self.vertices = []
        self.faces = []
        self.initial_v = None # For the triangulation, this gets set to the SW vertex on the bounding box
        self.N_vertices = 0
        
    def __iter__(self):
        assert self.initial_v is not None
        v = self.initial_v
        adj_edges = v.inc_edges
        for adj_e in adj_edges:
            w = adj_e.org.pos
            z = adj_e.dest().pos
            if (w[0] == z[0]):
                self.e = adj_e # v is only incident to an up edge, a right edge, and a NE edge.
                return self
    
    def __next__(self):
        e = self.e
        if e is None:
            raise StopIteration
        u = self.e.org.pos
        v = self.e.dest().pos
        adj_edges = self.e.dest().inc_edges
        is_up = (u[0] == v[0]) # the only choices are up, right, or SW
        
        if is_up:
            right_edge = None
            for adj_e in adj_edges:
                w = adj_e.org.pos
                z = adj_e.dest().pos
                if (w[0] == z[0] and w[1] < z[1]):
                    self.e = adj_e
                    return e
                elif right_edge is None and (w[1] == z[1] and w[0] < z[0]):
                    right_edge = adj_e
            # can only get to this part of the code below if there's no next up edge, i.e. if v is at the top of the lattice
            if right_edge is None:
                self.e = None
                return e
            else:
                self.e = right_edge
                return e
        is_right = (u[1] == v[1])
        if is_right:
            up_edge = None
            for adj_e in adj_edges:
                w = adj_e.org.pos
                z = adj_e.dest().pos
                if (z[0] < w[0] and z[1] < w[1]):
                    self.e = adj_e # next SW edge
                    return e
                elif up_edge is None and (w[0] == z[0] and w[1] < z[1]):
                    up_edge = adj_e
            # can only get to this part of the code below if there's no next SW edge, i.e. if v is at the bottom of the lattice
            self.e = up_edge
            return e
        else:
            # in this case, self.e must be SW and the next edge must be right
            for adj_e in adj_edges:
                w = adj_e.org.pos
                z = adj_e.dest().pos
                if (w[1] == z[1] and w[0] < z[0]):
                    self.e = adj_e
                    return e
            
    def __str__(self):
        # Only for testing
        str_ = "VERTICES\n"
        for v in self.vertices:
            str_ += (str(v) + "\n")
        str_ += "\nEDGES\n"
        for e in self.edges:
            str_ += (str(e) + "\n")
        str_ += "\nFACES\n"
        for f in self.faces:
            str_ += (str(f) + "\n")
        return str_
    
    def add_vertex(self, pos):
        v = Vertex(pos, self.N_vertices)
        self.N_vertices += 1
        self.vertices.append(v)
        return v
    
    def add_edge(self, v1, v2, spx_pair_list = None, T = None):
        '''
        v1, v2: Vertex objects adjacent to edge that we're adding.
        spx_pair_list: (optional) list of pairs of Simplex objects. The pairs of simplices that are swapping along the new edge.
        T: (optional) int. Index of triangle that new edge is within.
        
        Assumes both vertices are already added.
        Adds an edge between v1 and v2. Updates incident edges to v1 and v2.
        Does NOT update faces or next/prev edges.
        e1: v1-->v2, e2: v2-->v1
        '''
        e1 = Edge(v1)
        e2 = Edge(v2)
        e1.twin = e2
        e2.twin = e1
        e1.T = T
        e2.T = T
        self.edges.append(e1)
        self.edges.append(e2)
        v1.inc_edges.append(e1)
        v2.inc_edges.append(e2)
        
        if spx_pair_list is not None:
            e1.spx_pair = SpxPairList(spx_pair_list)
            e2.spx_pair = SpxPairList(spx_pair_list, forward = False)
        
        return e1, e2
    
    def add_face(self, edges):
        # edges: list of half edges on boundary of face, in counterclockwise order.
        # Assumes that all edges are already added.
        inc_edge = edges[0]
        f = Face(inc_edge)
        N = len(edges)
        for i, e in enumerate(edges):
            e.left = f
            e.next = edges[(i+1)%N]
            e.prev = edges[(i-1)%N]
        self.faces.append(f)
        return f
    
    def add_line(self, u, v, T, spx_pair = None):
        '''
        u, v: Vertex objects that are already added to the DCEL 
                (i.e., they're in self.vertices)
        T: index of the triangle that the line goes through.
        spx_pair: (optional) list of tuples of Simplex objects. The simplices that are swapping along this line.
            
        Adds the line (u, v) to DCEL by calculating intersections with the edges
        already in T and adding the appropriate new edges/vertices.
        '''
        e = u.get_inc_edge(T, v)
        #print("initial e = ", e.org.pos, e.dest().pos)
        w = e.dest()
        start_face = True # I think this stores whether we're *just starting a new face*, not whether we're in the initial face
        f = e.left
        split_face_edges = []
        while w != v:
            if start_face:
                #print("we're in the start face")
                start_face = False
                split_face_edges.append(e)
                e = e.next
                #print("e = ", e.org.pos, e.dest().pos)
            else:
                inter = DCEL.intersection(u.pos, v.pos, e.org.pos, w.pos)
                if inter is None:
                    #print("inter is None")
                    split_face_edges.append(e)
                    e = e.next
                    #print("e = ", e.org.pos, e.dest().pos)
                else:
                    #print("inter is not None")
                    if inter == w.pos:
                        #print("intersects exactly at a vertex")
                        # intersects exactly at vertex
                        split_face_edges.append(e)
                        
                        # find next face and next edge
                        f_count, e_count = self.counterclockwise_face(e)
                        f_clock, e_clock = self.clockwise_face(e)
                        #print("counter edge:", e_count.org.pos, e_count.dest().pos)
                        #print("clock edge: ", e_clock.org.pos, e_clock.dest().pos)
                        while f_count != f_clock:
                            f_count, e_count = self.counterclockwise_face(e_count)
                            f_clock, e_clock = self.clockwise_face(e_clock)
                        f_next = f_count # or f_clock, they're the same
                        e = e_count.next # or e_clock.prev, they're the same
                        #print("e = ", e.org.pos, e.dest().pos)
                        
                        # split face and reset for the next face
                        self.split_face(f, split_face_edges, spx_pair, T)
                        #print("split face")
                        split_face_edges = []
                        start_face = True
                        f = f_next
                        
                    else:
                        #print("doesn't intersect exactly at vertex")
                        self.split_edge(e, inter)
                        split_face_edges.append(e)
                        self.split_face(f, split_face_edges, spx_pair, T)
                        #print("split face")
                        e = e.twin
                        #print("e = ", e.org.pos, e.dest().pos)
                        start_face = True
                        f = e.left
                        split_face_edges = []
            w = e.dest()
        split_face_edges.append(e)
        self.split_face(f, split_face_edges, spx_pair, T)
        
    
    def clockwise_face(self, e):
        '''
        e: half edge that is incident to a face f
        
        Returns:
            f_: the face that is clockwise from f, as oriented around v = e.dest()
            e_: half edge that is incident to f_ and also has e_.dest() = v
        '''
        e_ = e.twin.prev
        if e_ is not None:
            f_ = e_.left
        else:
            f_ = None
        return f_, e_
    
    def counterclockwise_face(self, e):
        '''
        e: half edge that is incident to a face f
        
        Returns:
            f_: the face that is counterclockwise from f, as oriented around v = e.dest()
            e_: half edge that is incident to f_ and also has e_.dest() = v
        '''
        e_ = e.next.twin
        if e_ is not None:
            f_ = e_.left
        else:
            f_ = None
        return f_, e_
    
    def _update_dictionaries(T, e, w, spx1, spx2, D1, D2, D3):
        pair = frozenset({spx1, spx2})
        
        # Update D1
        if pair in D1[T]:
            '''
            w_ = D1[T][pair][0] # the other endpoint that we already found
            del D1[T][pair] # Remove the pair because we've now found its whole line segment
            finished_line_seg = True
            '''
            
            w_ = D1[T][pair][0] # the other endpoint that we already found
            D1[T][pair].append(w)
            finished_line_seg = True
            
        else:
            D1[T][pair] = [w]
            finished_line_seg = False # we've only found one endpoint of the line segment so far
            
        # Update D2 and D3 (assumes that swapping lines aren't edges of B!)
        if not finished_line_seg:
            # Update D3 because we haven't finished finding the whole line segment yet
            if w in D3[T]:
                D3[T][w].append({spx1, spx2})
            else:
                D3[T][w] = [{spx1, spx2}]
        else:
            # Update D3 because now we've found the whole line segment
            assert w_ in D3[T], "other vertex w_ should be in D3 but isn't"
            D3[T][w_].remove({spx1, spx2})
            if len(D3[T][w_]) == 0:
                del D3[T][w_]
            
            # Update D2 (keys are now ORDERED pairs of vertices to keep track of orientation)
            if e.T == T:
                # Then we're planesweeping this triangle in counterclockwise order, so the "forwards" half edge should be (w, w_), which is opposite order that we found them in
                vertex_pair = (w, w_)
            else:
                # Then we're planesweeping this triangle in clockwise order, so the "forwards" half edge should be (w_, w), which is the order that we found them in
                vertex_pair = (w_, w)
                
            if vertex_pair in D2[T]:
                D2[T][vertex_pair].append({spx1, spx2})
            else:
                D2[T][vertex_pair] = [{spx1, spx2}]
        #return D1, D2, D3
    
    def construct(simplices, xs, ys, spx_fcn = None, debug_mode = False, draw_mode = False):
        '''
        simplices: list of Simplex objects, not in any particular order.
        xs: list of floats. coordinates for 1st parameter
        ys: list of floats. coordinates for 2nd parameter
        
        
        Returns
        -----------
        dcel: DCEL object that represents the line arrangement induced by this 2-parameter family of filtered complexes.
        simplices: list of simplices, ordered by filtration at xs[-1], ys[-1]
        triang: matplotlib.tri.Triangulation object representing the initial triangulation.
        '''
        def check_simplices(idx):
            print("Checking order")
            correct_order = True
            for i, spx in enumerate(simplices[:-1]):
                print(spx.nodes, float(spx.val[idx]))
                correct_order = correct_order and (spx.val[idx] <= simplices[i+1].val[idx])
                if not correct_order:
                    print(simplices[i+1].nodes, float(simplices[i+1].val[idx]))
                    break
            if correct_order:
                print(simplices[-1].nodes, float(simplices[-1].val[idx]))
                print("\n")
            assert correct_order
        
        dcel = DCEL.triangulation(xs, ys) # initialize triangulation
        
        # Make matplotlib.tri.Triangulation object to represent triangulation
        xx = [v.pos[0] for v in dcel.vertices]
        yy = [v.pos[1] for v in dcel.vertices]
        triangles = []
        for f in dcel.faces:
            Tvs = f.vertices() # list of vertex coordinates for the triangle (face f)
            Tis = [v.idx for v in Tvs]
            triangles.append(Tis)
        triang = matplotlib.tri.Triangulation(xx, yy, triangles)
        
        if draw_mode:
            dcel.draw()
            
        num_T = (len(xs) - 1)*(len(ys) - 1)*2 # number of triangles
        D1 = [{} for i in range(num_T)]
        D2 = [{} for i in range(num_T)]
        D3 = [{} for i in range(num_T)]
        
        simplices.sort()
        for i, spx in enumerate(simplices):
            spx.curr_idx = i
            
        # Find all vertices that are on the 1-skeleton of the triangulation
        idx2 = (0, 0)
        for e in dcel:
            print("-----------------------------")
            Ts = [e.T, e.twin.T] # it's possible for one of these to be None
            curr_e = e
            u = e.org.pos
            v = e.dest().pos
            
            print("Edge endpoints: ", u, v)

            # Update indices of u and v
            idx1 = idx2 # index of u (origin of e) is the index of previous edge's destination
            x1, y1 = u
            x2, y2 = v
            if x2 > x1:
                h_offset = 1
            elif x2 < x1:
                h_offset = -1
            else:
                h_offset = 0
            if y2 > y1:
                v_offset = 1
            elif y2 < y1:
                v_offset = -1
            else:
                v_offset = 0
            idx2 = (idx2[0] + h_offset, idx2[1] + v_offset) # index of v (destination of e)
            
            # if debug_mode:
            #     # check order
            #     check_simplices(idx1)
            
            # Initialize the planesweep on edge e
            peq = PriorityQueue()
            prev_pos = u
            prev_dcel_vertex = e.org
            
            # DELETE THIS WHEN DONE DEBUGGING--- check order ---SLOWS DOWN THE PLANESWEEP
            check_simplices(idx1)
            '''
            sorted_simplices = sorted(simplices, key = lambda spx : spx.val[idx1])
            for spx, spx_ in zip(simplices, sorted_simplices):
                assert spx.nodes == spx_.nodes
            '''
            
            # Get intersections at initial vertex
            pairs = get_intersection_pairs_at_vertex(simplices, idx1, spx_fcn)
            if len(pairs) == 0: print("No nontrivial intersections at initial vertex")
            other_indices = get_third_vertices_indices(idx1, idx2, e)
            
            # Check which of those intersections are the endpoint of a line segment within a triangle (not an edge of B)
            for pair in pairs:
                spx1, spx2 = pair
                # First check that they're not equal along the edge we're currently sweeping
                if spx1.val[idx2] != spx2.val[idx2]:
                    # Then check each T to see if there's a line segment
                    for i, T in enumerate(Ts):
                        if T is not None:
                            idx3 = other_indices[i] # index of third vertex (not u or v) of triangle T
                            if spx1.val[idx3] != spx2.val[idx3]:
                                DCEL._update_dictionaries(T, e, e.org, spx1, spx2, D1, D2, D3)
                    
            
            # Check for initial crossings (previously, used to be handled as ActivateEvents)
            for spx in simplices:
                if spx.curr_idx > 0:
                    lower_nbr = simplices[spx.curr_idx - 1]
                    inter = spx.intersection(lower_nbr, 0, idx1, idx2)
                    if inter is not None:
                        t = inter[0]
                        y = inter[1]
                        if t == 0:
                            if spx.val[idx2] < lower_nbr.val[idx2]:
                                #print("Adding initial crossing with pairs: ", lower_nbr.nodes, spx.nodes)
                                peq.put(Crossing(lower_nbr, spx, t, y))
                            '''
                            else:
                                print("There's an initial intersection with pairs: ", lower_nbr.nodes, spx.nodes)
                                print("Not adding it because they're in right order wrt end of interval values")
                            '''
                        else:
                            #print("Adding initial crossing with pairs: ", lower_nbr.nodes, spx.nodes)
                            peq.put(Crossing(lower_nbr, spx, t, y))
            
            # Find and handle all crossings
            while not peq.empty():
                #print("\n")
                ev = peq.get()
                spx1 = ev.spx1
                spx2 = ev.spx2
                i = spx1.curr_idx
                #print("Crossing event: ", spx1.nodes, spx2.nodes)
                if i == spx2.curr_idx - 1:
                    #print("Indices are adjacent and in the right order (i.e., spx1.idx == spx2.idx - 1)")
                    # Swap the order of the simplices
                    simplices[i].curr_idx = i+1
                    simplices[i+1].curr_idx = i
                    simplices[i+1], simplices[i] = simplices[i], simplices[i+1]
                    
                    # Check if the swap is occuring in the same position as the last swap
                    pos = ((1 - ev.t)*u[0] + ev.t*v[0], (1 - ev.t)*u[1] + ev.t*v[1])
                    if pos != prev_pos:
                        #print("new swap is at new position: ", pos)
                        # Split curr_e at intersection and update curr_e
                        w = dcel.split_edge(curr_e, pos)
                        curr_e = w.inc_edges[1]
                        
                        # update variables
                        prev_pos = pos
                        prev_dcel_vertex = w
                    else:
                        #print("new swap is at previous vertex with position: ", prev_pos)
                        w = prev_dcel_vertex
                    
                    #pair = frozenset({spx1, spx2})
                    if ev.t != 0:
                        for T in Ts:
                            if T is not None:
                                DCEL._update_dictionaries(T, e, w, spx1, spx2, D1, D2, D3)
                    
                    # Check for new crossing events with new neighbors. 
                    # spx1 checks its new upper neighbor, and spx2 checks its 
                    # new lower neighbor.
                    if i + 2 < len(simplices):
                        spx1_up_nbr = simplices[i + 2]
                        spx1_up_intersect = spx1.intersection(spx1_up_nbr, ev.t, idx1, idx2)
                        if (spx1_up_intersect is not None) and (spx1.val[idx2] > spx1_up_nbr.val[idx2]):
                            # there's an intersection AND (this can happen if the intersection is at current time) they're actually changing order at current time (as opposed to detecting a swap we've already done)
                            peq.put(Crossing(spx1, spx1_up_nbr, spx1_up_intersect[0], spx1_up_intersect[1]))
                            #print("Adding new CE with pairs: ", spx1.nodes, spx1_up_nbr.nodes)
                    if i - 1 >= 0:
                        spx2_low_nbr = simplices[i - 1]
                        spx2_low_intersect = spx2.intersection(spx2_low_nbr, ev.t, idx1, idx2)
                        if (spx2_low_intersect is not None) and (spx2.val[idx2] < spx2_low_nbr.val[idx2]):
                            peq.put(Crossing(spx2_low_nbr, spx2, spx2_low_intersect[0], spx2_low_intersect[1]))
                            #print("Adding new CE with pairs: ", spx2_low_nbr.nodes, spx2.nodes)
        # debugging
        print("-----------------------------------------")
        print("Finished planesweep")
        print("Number of vertices in DCEL: ", len(dcel.vertices))
        
        # DELETE WHEN DONE DEBUGGING---THIS SLOWS THINGS DOWN
        check_simplices(idx2)
        '''
        print("Final order (via planesweep): ")
        for spx in simplices:
            print(spx.nodes)
        print("\n") 
        print("Filtration values (ground truth): ")
        sorted_simplices = sorted(simplices, key = lambda spx : spx.val[idx2])
        for spx in sorted_simplices:
            print(spx.nodes, float(spx.val[idx2]))
        '''            
        
        
        # debugging--- check the line segment dictionaries
        '''
        print("------------------------")
        print("Check D2 dictionary")
        for T, T_dict in enumerate(D2):
            print("\n")
            print("Triangle ", T)
            for vertex_pair, spx_pair_list in T_dict.items():
                #v0, v1 = vertex_pair
                #print("Will add line segment with vertices ", v0.pos, " and ", v1.pos)
                #print("Pairs: ")
                print("Will add line segment for pairs: ")
                for pair in spx_pair_list:
                    spx1, spx2 = pair
                    if (spx1.nodes == (97, ) or spx1.nodes == (99, ) ) and ( spx2.nodes == (97, ) or spx2.nodes == (99, )):
                        print(spx1.nodes, spx2.nodes)
        '''
                
           
        print("------------------------")

        # Add the line segments to the triangulation
        print("Adding the line segments to the triangulation")
        print("Number of triangles: ", len(D2))
        for T, T_dict in enumerate(D2):
            #print("\n")
            print("Triangle ", T, " : ", len(T_dict), " lines to add")
            i = 0
            
            # DEBUGGING BLOCK---CHECK THAT D1 IS EMPTY
            '''
            if D1[T]:
                print("D1 isn't empty")
                for spx_pair, _ in D1[T].items():
                    spx1, spx2 = spx_pair
                    print(spx1.nodes)
                    print(spx2.nodes)
                    print("\n")
            #assert not D1[T]
            '''
                
            for vertex_pair, spx_pair_list in T_dict.items():
                if i % 10 == 0: print(i)
                v0, v1 = vertex_pair
                #print("Adding line segment with vertices ", float(v0.pos[0]), float(v0.pos[1]), " and ", float(v1.pos[0]), float(v1.pos[1]))
                dcel.add_line(v0, v1, T, spx_pair_list)
                #dcel.draw()
                i += 1
              
        return dcel, simplices, triang
                    
    def draw(self, ignore_edges = False, ignore_vertices = True):
        plt.axis('equal')
        plt.axis('off')
        for f in self.faces:
            vertices = f.vertices()
            xs = [v.pos[0] for v in vertices]
            ys = [v.pos[1] for v in vertices]
            plt.fill(xs, ys, zorder = 1)
        
        if not ignore_vertices:
            for v in self.vertices:
                plt.scatter([v.pos[0]], [v.pos[1]], s = 100, zorder = 3)
            
        if not ignore_edges:
            for e in self.edges:
                x = float(e.org.pos[0])
                y = float(e.org.pos[1])
                #dx = float(e.dest().pos[0]) - x
                #dy = float(e.dest().pos[1]) - y

                plt.plot([x, e.dest().pos[0]], [y, e.dest().pos[1]], color = 'k', zorder = 2)
                #plt.arrow(x, y, dx, dy, color = 'k', width = .01, length_includes_head = True, zorder = 3)
        plt.show()
        
    def draw_face(f):
        plt.axis('equal')
        vertices = f.vertices()
        xs = [v.pos[0] for v in vertices]
        ys = [v.pos[1] for v in vertices]
        plt.fill(xs, ys)
            
    def dual_spanning_tree(self):
        '''     
        Returns 
        ----------------
        G: networkx Graph (directed). rooted spanning tree graph for dual graph to the dcel.
                    The root face is one of the faces incident to the dcel root vertex (NE vertex of bounding box)
                    Edges are directed AWAY from root.
            Vertices of the graph are Face objects.
            Edges of the graph correspond to half edges in the dcel. Each edge has an attribute that stores a reference to the dcel half edge.
        root: vertex in G. The root of the tree.
        '''
        # initialize tree
        self.T = nx.DiGraph()
        for e in self.root_vertex.inc_edges:
            if e.left is None:
                root_face = e.twin.left
                break
        '''
        for e in self.root_vertex.inc_edges:
            face = e.left
            if face is not None:
                root_face = face
                break
        '''
        self.T.add_node(root_face)
        
        # Construct tree thru DFS
        self.to_visit = queue.Queue()
        self.to_visit.put(root_face)
        while not self.to_visit.empty():
            self.dual_spanning_tree_add_nbrs(self.to_visit.get())
        return self.T, root_face
        
    def dual_spanning_tree_add_nbrs(self, face):
        # add unvisited nbrs
        edges = face.edges()
        for e in edges:
            adj_face = e.twin.left
            if (adj_face is not None) and not (self.T.has_node(adj_face)):
                self.T.add_edge(face, adj_face, dcel_edge = e) # this automatically adds the nbring node! the dcel edge is the half edge adj to "face", not the one adj to "adj_face"
                self.to_visit.put(adj_face)
        
    def intersection(a, b, c, d):
        '''
        a, b, c, d: Tuples.
        Returns the intersection (a tuple) between line segments (a, b) 
        and (c, d), if it exists. Assumes that the line segments do not overlap.
        
        If a, b, c, d are given as Fraction objects, then the exact intersection (as a Fraction object) is returned.
        '''
        line1vert = (a[0] == b[0])
        line2vert = (c[0] == d[0])
        if not line1vert:
            m1 = (b[1] - a[1])/(b[0] - a[0])
        if not line2vert:
            m2 = (d[1] - c[1])/(d[0] - c[0])
            
        if line1vert and line2vert:
            return None
        if line1vert:
            if min(c[0], d[0]) <= a[0] <= max(c[0], d[0]):
                x_ = a[0]
                y_ = m2*(x_ - c[0]) + c[1]
                if min(a[1], b[1]) <= y_ <= max(a[1], b[1]):
                    return (x_, y_)
                else: return None
            else: return None
        if line2vert:
            if min(a[0], b[0]) <= c[0] <= max(b[0], d[0]):
                x_ = c[0]
                y_ = m1*(x_ - a[0]) + a[1]
                if min(c[1], d[1]) <= y_ <= max(c[1], d[1]):
                    return (x_, y_)
                else: return None
            else: return None
        
        else:
            if m1 == m2:
                return None
            else:
                x_ = (c[1] - c[0]*m2 + a[0]*m1 - a[1])/(m1 - m2)
                if max(min(a[0], b[0]), min(c[0], d[0])) <= x_ <= min(max(a[0], b[0]), max(c[0], d[0])):
                    y_ = m1*(x_ - a[0]) + a[1]
                    if max(min(a[1], b[1]), min(c[1], d[1])) <= y_ <= min(max(a[1], b[1]), max(c[1], d[1])):
                        return (x_, y_)
                    else: return None
                else:
                    return None
            
    def split_edge(self, e, pos):
        '''
        e: Edge object. edge to split.
        pos: position of new vertex to add in the middle of e
        
        e gets split into two edges. e becomes one of those two.
        Returns the new Vertex v.
        '''
        v = self.add_vertex(pos)
        u = e.dest()
        
        # reset destination of e
        e.twin.org = v
        v.inc_edges.append(e.twin)
        u.inc_edges.remove(e.twin)
        
        # add new edge, set incident faces, incident triangles, and spx_pairs
        e1, e2 = self.add_edge(v, u)
        e1.left = e.left
        e2.left = e.twin.left
        e1.T = e.T
        e2.T = e.twin.T
        e1.spx_pair = e.spx_pair
        e2.spx_pair = e.spx_pair
        
        # reset prev/next edges
        e3 = e.next
        e4 = e.twin.prev
        if e3 is not None:
            e3.prev = e1
            e1.next = e3
        if e4 is not None:
            e4.next = e2
            e2.prev = e4
        e1.prev = e
        e.next = e1
        e2.next = e.twin
        e.twin.prev = e2
        return v
        
    def split_face(self, f, edges, spx_pair = None, T = None):
        '''
        f: face to split
        edges: list of Edge objects (in counterclockwise order) 
        that will be incident to the new face (this is a subset of the edges on
                                               the boundary of f)
        spx_pair: (optional) tuple of Simplex objects. The pair of simplices that are swapping along the new edge.
        
        T: (optional) int. Index of triangle that the new edge is within.
        
        We add an edge between edges[0].org and edges[-1].dest.
        Assumes that the edge is in the interior of a triangle in the triangulation
        (except for possibly the vertices, which may be on the boundary of a triangle
         but cannot be any of the vertices of the triangle)
        
        Returns 
        e1: edges[0].org-->edges[-1].dest
        e2: edges[-1].dest-->edges[0].org
        '''
        # add edge, update inc_edge of f in case its previous inc_edge is going to be an edge of the new face
        u = edges[0].org
        v = edges[-1].twin.org
        e1, e2 = self.add_edge(u, v, spx_pair, T)
        f.inc_edge = e1
        
        # add face
        f2 = Face(e2)
        self.faces.append(f2)
        
        # Update incident faces
        e1.left = f
        e2.left = f2
        for e in edges:
            e.left = f2
        
        # Update next/prev edges
        e = edges[-1].next
        e.prev = e1
        e1.next = e
        
        e = edges[0].prev
        e.next = e1
        e1.prev = e
        
        e = edges[-1]
        e.next = e2
        e2.prev = e
        
        e = edges[0]
        e.prev = e2
        e2.next = e
        
        return e1, e2
        
    def triangulation(xs, ys):
        # Create bounding box
        dcel = DCEL()
        v1 = dcel.add_vertex((xs[0], ys[0]))
        dcel.initial_v = v1
        v2 = dcel.add_vertex((xs[-1], ys[0]))
        v3 = dcel.add_vertex((xs[-1], ys[-1])) # NE vertex
        v4 = dcel.add_vertex((xs[0], ys[-1]))
        e1, _ = dcel.add_edge(v1, v2)
        e2, _ = dcel.add_edge(v2, v3)
        e3, _ = dcel.add_edge(v3, v4)
        e4, _ = dcel.add_edge(v4, v1)
        dcel.add_face([e1, e2, e3, e4])
        
        # set root vertex
        dcel.root_vertex = v3
        
        # Add vertical edges
        y_top = ys[-1]
        y_bot = ys[0]
        def add_vertical_edge(left_e, x):
            # Add vertical edge with given x coordinate + vertical edge to its left
            # Returns: The half edge of the new edge that points downwards
            top = left_e.prev
            bot = left_e.next
            f = left_e.left
            dcel.split_edge(top, (x, y_top))
            dcel.split_edge(bot, (x, y_bot))
            new_e, _ = dcel.split_face(f, [left_e.prev, left_e, left_e.next])
            return new_e
        left_e = e4
        for x in xs[1:-1]:
            left_e = add_vertical_edge(left_e, x)
            
        # Add horizontal lines
        def add_horizontal_line(left_e, y):
            # Add horizontal line with given y coordinate. left_e is the leftmost edge to split (oriented downwards)
            # Returns: The half edge for the left_e for the next horizontal line to split
            x = left_e.org.pos[0]
            dcel.split_edge(left_e, (x, y)) # when we split, left_e becomes the half edge that's above
            e = left_e.next
            f = e.left
            while f is not None:
                e_bot = e.next
                e_right = e_bot.next
                x = e_right.org.pos[0]
                dcel.split_edge(e_right, (x, y))
                dcel.split_face(f, [e, e_bot, e_right])
                e = e_right.twin
                f = e.left
            return left_e
        left_e = e4
        for y in ys[1:-1]:
            left_e = add_horizontal_line(left_e, y)
            
        # Add diagonal edges
        faces = dcel.faces.copy()
        for f in faces:
            dcel.split_face(f, [f.inc_edge, f.inc_edge.next])
            
        # Assign a number to each triangle, and label each half edge with the number of the triangle it's adjacent to
        i = 0
        for f in dcel.faces:
            for e in f.edges():
                e.T = i
            i += 1
        return dcel
    
    def tri_polygons(self):
        '''

        Returns
        -------
        triang: a matplotlib.tri.Triangulation object whose vertices are the vertices of the dcel. We triangulate each polygon (face) of the DCEL.

        '''
        x = [v.pos[0] for v in self.vertices]
        y = [v.pos[1] for v in self.vertices]
        triangles = []
        for f in self.faces:
            v_indices = [v.idx for v in f.vertices()]
            Nsides = len(v_indices)
            for i in range(1, Nsides - 1):
                triangles.append([v_indices[0], v_indices[i], v_indices[i+1]])
        triang = matplotlib.tri.Triangulation(x, y, triangles)
        return triang
    
##############################################################################
class SpxPairList:
    def __init__(self, init_spx_pairs, forward = True):
        '''
        init_spx_pairs: list of simplex pairs to initialize list.
        forward: Boolean (default True). True if the list is in order, false, if you want to pop off elements in reverse order
        '''
        self.list = init_spx_pairs
        self.forward = forward
        
    def __iter__(self):
        if self.forward:
            self.curr_idx = 0
            self.orient = 1
        else:
            self.curr_idx = len(self.list) - 1
            self.orient = -1
        self.search_idx = self.curr_idx
        return self
    
    def __next__(self):
        # Returns the next simplex pair in list (according to orientation, i.e. self.forward) s.t. the indices are adjacent.
        # DOES NOT UPDATE THE SIMPLEX INDICES
        # Then update curr_idx, next_idx
        # If the immediate next element on list doesn't satisfy index condition, then move it back one spot.
        if (self.forward and self.curr_idx < len(self.list)) or (not self.forward and self.curr_idx >= 0):
            pair = self.list[self.curr_idx]
            spx1, spx2  = pair
            
            # Check if it's ready for an RU swap, and if not, try the next pair.
            while abs(spx1.curr_idx - spx2.curr_idx) > 1:
                # increment search index
                print("Pair wasn't ready for RU update")
                self.search_idx += self.orient
                
                # try the next pair
                pair = self.list[self.search_idx]
                spx1, spx2 = pair
                
            # Now move the pair to curr_idx, if it isn't already there.
            # If there are a lot of pairs and we often have to do this, then this would be way more efficient with linked list, but I don't expect us to be in that situation.
            if self.search_idx != self.curr_idx:
                if self.forward:
                    self.list.insert(self.curr_idx, pair)
                    del self.list[self.search_idx + 1]
                else:
                    self.list.insert(self.curr_idx + 1, pair)
                    del self.list[self.search_idx]
            
            # increment curr_idx and and reset search_idx
            self.curr_idx += self.orient
            self.search_idx = self.curr_idx
            return pair
        else:
            raise StopIteration
        
    def append(self, spx_pair):
        self.list.append(spx_pair)
        
        
##############################################################################
class PDB:
    def __init__(self, simplices, xs, ys, spx_fcn = None):
        '''
        simplices: list of Simplex objects, not in any particular order
        xs: list of floats. coordinates for 1st non-filtration parameter, evenly spaced
        ys: list of floats. coordinates for 2nd non-filtration parameter, evenly spaced
        '''
        self.xs = xs
        self.ys = ys
        #self.simplices = simplices
        dx = xs[1] - xs[0] # Assumes that dx = xs[i+1] - xs[i] for all i
        dy = ys[1] - ys[0] # Assumes that dy = ys[j+1] - ys[j] for all j
        
        # construct DCEL. Afterwards, self.simplices will now be ordered by filtration at xs[-1], ys[-1] (NE corner of domain grid)
        self.dcel, simplices, self.triang = DCEL.construct(simplices, xs, ys, spx_fcn)  # self.triang is the original triangulation of B

        Is = [round((x - xs[0])/dx) for x in self.triang.x] 
        Js = [round((y - ys[0])/dy) for y in self.triang.y]
        N = len(Is)
        
        print("\n")
        print("-----------------------------------------")
        print("\n")
        print("Calculating pairs")
        print("\n")
        for spx in simplices:
            spx.z = [float(spx.val[Is[k], Js[k]]) for k in range(N)]
            spx.interpolator = matplotlib.tri.LinearTriInterpolator(self.triang, spx.z)
        
        PDB.calculate_pairs(self.dcel, simplices)
        self.dcel_triang = None # This is a triangulation of the polygons in the DCEL, used for plotting persistence functions
        self.tri_finder = None
        
    def calculate_pairs(dcel, simplices):
            
        # Calculate path through faces of the dcel
        T, root = dcel.dual_spanning_tree() # T is a graph where the vertices of T are faces of the dcel, and edges have an attribute 'dcel_edge'
        path = PDB.path_thru_tree(T, root)
        print("Calculated path through dcel")
            
        # Initialize RU decomposition
        ru = RU(simplices) # RU decomposition for the root face (one of the faces adjacent to xs[-1], ys[-1])
        
        # initialize pairs
        pair_idx_dict = ru.get_spx_pairs() # indices of the pairs for root face
        pairs = []
        for birth_idx, death_idx in pair_idx_dict.items():
            if death_idx is None:
                pairs.append([simplices[birth_idx], None])
            else:
                pairs.append([simplices[birth_idx], simplices[death_idx]])
            
        # initialize spx_to_pair dictionary
        spx_to_pair = {} # key is spx, value is (i, j) s.t. pairs[i][j] = spx
        for i, pair in enumerate(pairs):
            spx_b = pair[0]
            spx_to_pair.update({spx_b : (i, 0)})
            spx_d = pair[1]
            if spx_d is not None:
                spx_to_pair.update({spx_d : (i, 1)})
        root.spx_pairs = pairs
        prev_face = root
        
        print("Initialized everything")
        
        num_faces = len(path)
        for i, face in enumerate(path[1:]):
            # update RU decomposition and pairs
            print(f"FACE {i + 2} / {num_faces}")
                      
            if T.has_edge(prev_face, face):
                dcel_edge = T[prev_face][face]['dcel_edge']
            else:
                dcel_edge = T[face][prev_face]['dcel_edge'].twin
            swapping_pairs = dcel_edge.spx_pair # list of pairs to swap
        
            if swapping_pairs is not None:
                any_updates = False
                
                for spx_pair in iter(swapping_pairs):
                    spx1, spx2 = spx_pair # pair of simplices that we're swapping as we move to the new face
                    print(spx1.nodes, spx2.nodes)
                    spx1.curr_idx, spx2.curr_idx = spx2.curr_idx, spx1.curr_idx # swap the indices that the simplices store
                    
                    need_update = ru.swap(spx1, spx2) # update the RU decomposition and return whether we need to update the (birth, death) simplex pairs
                    
                    if need_update:
                        if not any_updates:
                            any_updates = True
                            pairs2 = []
                            for pair in pairs:
                                spxb, spxd = pair
                                pairs2.append([spxb, spxd])
                            
                        i1 = spx_to_pair[spx1][0]
                        i2 = spx_to_pair[spx2][0]
                        j1 = spx_to_pair[spx1][1]
                        j2 = spx_to_pair[spx2][1]
                        
                        # update/store pairs---swap spx1 and spx2 within the pairs that contain them. Everything else stays the same
                        pairs2[i1][j1] = spx2
                        pairs2[i2][j2] = spx1
                        
                        # update spx_to_pairs dictionary
                        spx_to_pair.update({spx1 : (i2, j2)})
                        spx_to_pair.update({spx2 : (i1, j1)})
                        
                if any_updates: 
                    face.spx_pairs = pairs2
                else:
                    # There were swapping pairs, but none caused any PH-pair updates
                    # TO DO- delete the edge in the DCEL that's adjacent to face and prev_face (the edge is stored in dcel_edge)
                    face.spx_pairs = pairs
            
            else:
                # No swapping pairs
                # TO DO- delete the edge in the DCEL that's adjacent to face and prev_face (the edge is stored in dcel_edge)
                face.spx_pairs = pairs
                
            pairs = face.spx_pairs
            prev_face = face
    
    def get_data(self, fn, num_bins = 15, **kwargs):
        '''
        Parameters
        ----------
        fn : callable. 
            A function whose input is self and a function of p in B.
            Isn't necessarily a real-valued function (e.g., fn = query_ph)
        num_bins : int (positive)
        **kwargs: variables that are inputs to fn.
        '''
        x0 = float(self.xs[0])
        xN = float(self.xs[-1])
        dx = (xN - x0)/num_bins
        
        y0 = float(self.ys[0])
        yN = float(self.ys[-1])
        dy = (yN - y0)/num_bins
        
        Xs = np.arange(x0, xN + dx, dx)
        if len(Xs) > num_bins + 1:
            Xs = Xs[:-1] # it should've been length (num_bins) in the first place, but floating point error could happen
        Xs[-1] = xN # should be xN anyway, but there could be floating point error
        
        Ys = np.arange(y0, yN + dy, dy)
        if len(Ys) > num_bins + 1:
            Ys = Ys[:-1]
        Ys[-1] = yN
        
        Xs, Ys = np.meshgrid(Xs, Ys)
        Zs = {}
        for i in range(num_bins + 1):
            for j in range(num_bins + 1):
                Zs[(i, j)] = fn(self, Xs[i, j], Ys[i, j], **kwargs)
        return Xs, Ys, Zs
        
    def path_thru_tree(G, root):
        '''
        G: networkx directed tree graph where the vertices are Face objects.
            Use case is for when G represents spanning tree for the dual graph to a DCEL.
        root: node in G to start at.
        
        Returns
        full_path: list of nodes in G. A path through the tree G.
        '''
        curr_node = root
        #print(curr_node.inc_edge)
        visited_nodes = {curr_node}
        full_path = [curr_node]
        
        found_next_node = False
        unvisited_nbr = None
        path = [] # path to curr_node from from earliest branch point that still has unexplored directions
        for inc_edge in G.edges(curr_node):
            nbr = inc_edge[1]
            if not nbr in visited_nodes:
                if not found_next_node:
                    edge = inc_edge
                    unvisited_nbr = nbr
                    found_next_node = True
                else:
                    # if root has > 2 nbrs, then it's a branch point
                    path.append(edge)
                    break
        
        while unvisited_nbr is not None or len(path) > 0:
            if unvisited_nbr is not None:
                curr_node = unvisited_nbr
                #print(curr_node.inc_edge)
                visited_nodes.add(curr_node)
                full_path.append(curr_node)
                
                # look for next node
                found_next_node = False
                unvisited_nbr = None
                for inc_edge in G.edges(curr_node):
                    nbr = inc_edge[1]
                    if not nbr in visited_nodes:
                        if not found_next_node:
                            edge = inc_edge
                            unvisited_nbr = nbr
                            found_next_node = True
                            if len(path) > 0: 
                                # if we already know we need to backtrack later, then immediately add this edge to the path and break
                                path.append(edge)
                                break
                        else:
                            path.append(edge)
                            break
            else:
                while unvisited_nbr is None:
                    curr_node = path[-1][0]
                    full_path.append(curr_node)
                    #print(curr_node.inc_edge)
                    path.pop()
                    
                    # look for next node by checking nbrs of curr_node
                    for inc_edge in G.edges(curr_node):
                        nbr = inc_edge[1]
                        if not nbr in visited_nodes:
                            if unvisited_nbr is None:
                                edge = inc_edge
                                unvisited_nbr = nbr
                                if len(path) > 0:
                                    # if we already know we need to backtrack later, then immediately add this edge to the path and break
                                    path.append(edge)
                                    break
                            else:
                                path.append(edge)
                                break
        return full_path
    
    def ph_from_face_point(self, face, p):
        x, y = p
        pairs = face.spx_pairs
        ph = pairs_to_ph(pairs, x, y)
        return ph
    
    def dgm_from_face_point(self, face, p, dim = 0):
        ph = self.ph_from_face_point(face, p)
        dgm = ph_to_non_essential_dgm(ph, dim)
        return dgm
    
    def ph(self, v):
        '''
        Parameters
        ----------
        v : Vertex object.
            Vertex in self.dcel

        Returns
        -------
        ph: list. Each element is of form [dim, [b, d]]
            Persistent homology at v
        '''
        f = v.get_inc_face()
        x = float(v.pos[0])
        y = float(v.pos[1])
        pairs = f.spx_pairs
        ph = pairs_to_ph(pairs, x, y)
        return ph
                
    def plot_function(self, f, use_mesh = True, num_bins = 15, only_z = False, elev = 50, azim = -45, xlabel = 'X', ylabel = 'Y', zlabel = 'Z', **kwargs):
        '''
        f: real-valued function that takes ph (as returned by ph(v)) and **kwargs as input
        use_mesh: (boolean) Use grid points of B to plot the function. If false, use self.dcel.vertices.
        num_bins: (int) Number of bins for meshing, if use_mesh. Uses a (num_bins)x(num_bins) mesh.
        '''
        if use_mesh:
            Xs, Ys = generate_mesh(self.xs, self.ys, num_bins)
            z = np.zeros((num_bins + 1, num_bins + 1))
            for i in range(num_bins + 1):
                for j in range(num_bins + 1):
                    z[i, j] = f(self.query_ph(Xs[i, j], Ys[i, j]), **kwargs)
        else:
            # Triangulate each polygon in the dcel
            if self.dcel_triang is None:
                self.dcel_triang = self.dcel.tri_polygons()
            
            # Get the values of f at the vertices of the triangulation (which are ordered by their order in self.dcel.vertices)
            z = np.array([f(self.ph(v), **kwargs) for v in self.dcel.vertices])
            
        if not only_z:
            assert np.max(z) < np.inf, "Can't plot function with infinite values"
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            if use_mesh:
                ax.plot_surface(Xs, Ys, z, cmap= 'jet', linewidth = 0, antialiased = False)
            else:
                ax.plot_trisurf(self.dcel_triang, z, cmap = 'jet')
            ax.view_init(elev = elev, azim = azim)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.show()
            return ax
        
    def query_pairs(self, x, y):
        if self.tri_finder is None:
            if self.dcel_triang is None:
                self.dcel_triang = self.dcel.tri_polygons()
            self.tri_finder = self.dcel_triang.get_trifinder()
        tri_idx = self.tri_finder(x, y)
        tri_vertex_indices = self.dcel_triang.triangles[tri_idx, :]
        tri_vertices = [self.dcel.vertices[i] for i in tri_vertex_indices] # the three Vertex objects
        face_sets = [v.get_all_inc_faces() for v in tri_vertices]
        F = next(iter(face_sets[0].intersection(face_sets[1]).intersection(face_sets[2]))) # get single element in the intersection of all three face sets
        pairs = F.spx_pairs
        return pairs
        
    def query_ph(self, x, y):
        # x, y: floats. Coordinates for a point in base space B
        # Returns: ph at p
        #
        # uses triFinder, which implements trapezoid map algorithm from the book 
        # "Computational Geometry, Algorithms and Applications, second edition, 
        # by M. de Berg, M. van Kreveld, M. Overmars. (Source: https://github.com/matplotlib/matplotlib/blob/v3.8.1/lib/matplotlib/tri/_trifinder.py#L7-L24)
        # and O. Schwarzkopf.
        if self.tri_finder is None:
            if self.dcel_triang is None:
                self.dcel_triang = self.dcel.tri_polygons()
            self.tri_finder = self.dcel_triang.get_trifinder()
        tri_idx = self.tri_finder(x, y)
        tri_vertex_indices = self.dcel_triang.triangles[tri_idx, :]
        tri_vertices = [self.dcel.vertices[i] for i in tri_vertex_indices] # the three Vertex objects
        face_sets = [v.get_all_inc_faces() for v in tri_vertices]
        F = next(iter(face_sets[0].intersection(face_sets[1]).intersection(face_sets[2]))) # get single element in the intersection of all three face sets
        ph = self.ph_from_face_point(F, (x, y))
        return ph
    
    def query_dgm(self, x, y, dim = 0):
        ph = self.query_ph(x, y)
        dgm = ph_to_non_essential_dgm(ph, dim)
        return dgm

##########################################

def all_pairs_in_list(spx_list, spx_fcn = None):
    # given a list (of unique elements), return a list of all (unordered) pairs in list
    # spx_fcn is Callable whose signature is spx_fn(Simplex object) and returns Boolean value
    # The example in mind for spx_fcn: is_dim0 (Ignore pairs of 0-dimensional simplices, which often times have the same filtration value (zero) throughout the whole base space
    if spx_fcn is None:
        pairs = []
        for i, a in enumerate(spx_list):
            for b in spx_list[:i]:
                pairs.append((a, b))

    else:
        special_simplices = []
        other_simplices = []
        for spx in spx_list:
            if spx_fcn(spx):
                special_simplices.append(spx)
            else:
                other_simplices.append(spx)
        pairs = []
        
        # get bipartite pairs
        for a in special_simplices:
            for b in other_simplices:
                pairs.append((a, b))
        
        # get pairs within other_simplices
        for i, a in enumerate(other_simplices):
            for b in other_simplices[:i]:
                pairs.append((a, b))
            
    return pairs
    
def entropy(ph, dim):
    '''
    ph: list
    Persistent homology. Each element is of form [dim, [b, d]]
    dim: integer
    
    Returns:
        entropy: float
        Persistent entropy
    '''
    Ls = lifetimes(ph, dim, finite = True)
    S = np.sum(Ls)
    ps = Ls/S

    entropy = -np.sum([p*math.log(p) for p in ps])
    return entropy
        
def finite_total_persistence(ph, dim):
    '''
    ph: list
    Persistent homology. Each element is of form [dim, [b, d]]
    dim: integer
    
    Returns:
        TP: float
        Sum of lifetimes (d - b) for the (dim)-dimensional homology classes such that d != np.inf
    '''
    TP = np.sum(lifetimes(ph, dim, finite = True))
    return TP

def get_intersections_at_vertex(simplices, idx):
    # Returns dictionary where key = a certain filtration value, value of key = list of simplices with that viltration value at vertex in B with index idx
    D = {}
    for spx in simplices:
        filt_val = spx.val[idx]
        D.setdefault(filt_val, [])
        D[filt_val].append(spx)
    D_ = {}
    for key, val in D.items():
        if len(val) > 1:
            D_[key] = val
    return D_

def generate_mesh(xs, ys, num_bins):
    x0 = float(xs[0])
    xN = float(xs[-1])
    dx = (xN - x0)/num_bins
    
    y0 = float(ys[0])
    yN = float(ys[-1])
    dy = (yN - y0)/num_bins
    
    Xs = np.arange(x0, xN + dx, dx)
    if len(Xs) > num_bins + 1:
        Xs = Xs[:-1] # it should've been length (num_bins) in the first place, but floating point error could happen
    Xs[-1] = xN # should be xN anyway, but there could be floating point error
    Ys = np.arange(y0, yN + dy, dy)
    if len(Ys) > num_bins + 1:
        Ys = Ys[:-1]
    Ys[-1] = yN
    
    Xs, Ys = np.meshgrid(Xs, Ys)
    return Xs, Ys

def get_fn_vals(fn, PHs, num_bins, **kwargs):
    # PHs: dictionary from tuples (i, j)---indexing a mesh of base space---to PH
    # Returns: np.array whose (i, j)th entry is fn(PH(i, j))
    fn_vals = np.array([[fn(PHs[(i, j)], **kwargs) for j in range(num_bins + 1)] for i in range(num_bins + 1)])
    return fn_vals

def get_intersection_pairs_at_vertex(simplices, idx, spx_fcn = None):
    all_pairs = []
    D = get_intersections_at_vertex(simplices, idx)
    for key, val in D.items():
        pairs = all_pairs_in_list(val, spx_fcn)
        all_pairs += pairs
    return all_pairs   

def get_third_vertices_indices(idx1, idx2, e):
    vertical = (idx1[0] == idx2[0])
    horizontal = (idx1[1] == idx2[1])
    
    i, j = idx1
    if vertical:
        u1 = (i - 1, j)
        u2 = (i + 1, j + 1)
    elif horizontal:
        u1 = (i + 1, j + 1)
        u2 = (i, j - 1)
    else:
        # must be diagonal
        u1 = (i, j - 1)
        u2 = (i - 1, j)
        
    if e.left is None:
        u1 = None
    if e.twin.left is None:
        u2 = None
    
    us = [u1, u2]
    return us

def is_dim0(spx):
    return len(spx.nodes) == 1

def kth_longest_lifetime(ph, dim, k, finite = True):
    '''
    Parameters
    ----------
    ph : list where each element is of form [dim, [b, d]]
        Persistent homology
    dim : int
        homology dimension
    k : int (positive)

    Returns
    -------
    kthlife: float
        Length of the kth longest PH lifetime (among (dim)-dimensional classes). Could be np.inf
    '''
    Ls = lifetimes(ph, dim, finite)
    if k < len(Ls):
        kth_life = abs(np.partition(-Ls, k)[k])
        return kth_life
    return 0
        
def lifetimes(ph, dim, finite = False):
    if finite:
        lifetimes = np.array([elt[1][1] - elt[1][0] for elt in ph if elt[0] == dim and elt[1][1] < np.inf])
    else:
        lifetimes = np.array([elt[1][1] - elt[1][0] for elt in ph if elt[0] == dim])
    return lifetimes

def pairs_to_ph(pairs, x, y):
    ph = []
    for pair in pairs:
        spx_b, spx_d = pair
        dim = int(len(spx_b.nodes) - 1)
        b = float(spx_b(x, y))
        if spx_d is not None:
            d = float(spx_d(x, y))
        else:
            d = np.inf
        if b != d:
            ph.append([dim, [b, d]])
    return ph
    
def ph_to_non_essential_dgm(ph, dim = 0):
    dgm = np.array([[pt[1][0], pt[1][1]] for pt in ph if pt[1][1] != np.inf and pt[0] == dim])
    return dgm

def plot_data(xs, ys, Zs, num_bins, elev = 50, azim = -45, xlabel = 'X', ylabel = 'Y', zlabel = 'Z'):
    assert np.max(Zs) < np.inf
    Xs, Ys = generate_mesh(xs, ys, num_bins)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(Xs, Ys, Zs, cmap= 'jet', linewidth = 0, antialiased = False)
    ax.view_init(elev = elev, azim = azim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def wasserstein_distance_approx(PHs1, PHs2, xs, ys, num_bins, r = 2):
    assert r >= 1
    W_rpq = 0
    dx = (xs[-1] - xs[0])/num_bins
    dy = (ys[-1] - ys[0])/num_bins   
    for idx, ph in PHs1.items():
        dgm1 = ph_to_non_essential_dgm(ph)
        dgm2 = ph_to_non_essential_dgm(PHs2[idx])
        Wass_dist = W(dgm1, dgm2)
        if r == np.inf:
            W_rpq = max(W_rpq, Wass_dist)
        else:
            W_rpq += (Wass_dist**r)*dx*dy
    if r < np.inf:
        W_rpq = W_rpq**(1/r)
    return W_rpq
###############################
###############################