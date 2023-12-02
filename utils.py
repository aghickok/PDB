class Node:
    # In the vinyard context- node.data stores its row index in the original reduced matrix
    def __init__(self, data = None):
        self.data = data
        self.next = None
    
    def copy(node):
        copy_node = Node(node.data)
        copy_node.next = node.next
        return copy_node
    
    def print(self):
        print("Data value: ",self.data)
        if self.next is not None:
            print("Next node value: ", self.next.data)
        else:
            print("Next node: None")
            
class linkedList:
    # In the vineyard context- need to be able to add nodes to the end of the 
    # list, remove node with particular value, merge two lists (delete 
    # duplicate nodes), insert nodes at correct place, and check if any of the 
    # nodes have a particular value. The linked list must stay sorted by value, 
    # which represents a node's original row index.
    def __init__(self):
        self.head = None
        
    def __iadd__(self, other):
        other_curr = other.head
        while (other_curr is not None and (self.head is None or other_curr.data <= self.head.data)):
            if self.head is None or other_curr.data < self.head.data:
                old_head = self.head
                self.head = Node(other_curr.data)
                self.head.next = old_head
                other_curr = other_curr.next
            else:
                self.head = self.head.next
                other_curr = other_curr.next
        self_curr = self.head
            
        # Now self_curr is not None and other_curr.data > self_curr.data.
        # We're going to insert a copy of other_curr somewhere after self_curr, 
        # although maybe not immediately afterwards.
        while other_curr is not None:
            if self_curr.next is None or other_curr.data < self_curr.next.data:
                # insert a copy of other_curr immediately after self.curr
                new_node = Node(other_curr.data)
                new_node.next = self_curr.next
                self_curr.next = new_node
                other_curr = other_curr.next # increment other_curr
                if self_curr.next is not None:
                    self_curr= self_curr.next
            elif other_curr.data == self_curr.next.data:
                self_curr.next = self_curr.next.next # delete self_curr from self
                other_curr = other_curr.next # increment other_curr
            elif self_curr.next is not None:
                self_curr = self_curr.next # increment self_curr
        return self
                    
    def append(self, node):
        # Append node to the end of the linked list.
        # Seems to mostly (only?) get used in debugging/test functions.
        if self.head is None:
            self.head = node
        else:
            curr = self.head
            while(curr.next is not None):
                curr = curr.next
            curr.next = node
            
    def has(self, data_val):
        # Returns true if any of the nodes have data = data_val.
        curr = self.head
        while(curr is not None):
            if curr.data == data_val:
                return True
            curr = curr.next
        return False
    
    def insert(self, node):
        # Insert node at correct position to keep list sorted.
        # This gets used when constructing the initial boundary matrix.
        curr = self.head
        if curr is None or curr.data > node.data:
            node.next = self.head
            self.head = node
        else: 
            while(curr.next is not None and curr.next.data < node.data):
                curr = curr.next
            # Now either curr.next is None (we're at the end of the list), or 
            # curr.next is the first node whose value is >= node.data. In either
            # case, we want to insert node right after curr
            node.next = curr.next # could be None
            curr.next = node
        
    def print(self):
        curr = self.head
        while curr is not None:
            print(curr.data)
            curr = curr.next
    
    def remove(self, data_val):
        # remove node with node.data = data_val, if it's there.
        # Assume there's only one (for vineyard context, this is true)
        # Used for setting entries of a sparseMatrix equal to zero.
        if self.head is not None:
            curr = self.head
            if curr.data == data_val:
                self.head = curr.next
            else:
                while(curr.next is not None and not curr.next.data == data_val):
                    curr = curr.next
                # curr is now either the last node or the node right before the 
                # node whose data is data_val
                if not curr.next is None:
                    curr.next = curr.next.next
