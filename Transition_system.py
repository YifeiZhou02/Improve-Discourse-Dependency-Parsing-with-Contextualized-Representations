#the transition system
#an instance of transition system can be used only once
num_EDUs = 6
import copy
import numpy as np
import random
import torch

class Transition_system:
    """
    a superclass of the transition system
    an instance of transition system that can be used only once
    should create a new one for each execution
    """
    def __init__(self,EDUs):
        """
        create a transition system
        with default initial configuration
        """
        #the queue in the transition system
        self.stack = []
        self.queue = EDUs.copy()
        self.paragraph = EDUs.copy()
        #mapping from id to id
        self.heads = {}
        try:
            self.embeddings_dim = len(EDUs[0].embeddings)
        except TypeError:
             self.embeddings_dim = 0

    
    #return the feature description of the current state
    def get_feature(self):
        """
        return the feature representing the current state
        support options: 'str' and 'embeddings'
        str represent each EDU with its sentence
        embeddings represent each EDU with its arrtibute embeddings
        """
      #return the string representation of feature for str option
        bert_dim = self.embeddings_dim
        # feature = torch.cat([torch.Tensor(EMPTY) for i in range(num_EDUs)], dim = 0).reshape(1,bert_dim*num_EDUs)
        feature = torch.zeros(1,bert_dim*num_EDUs)
        #fill it with 0 if there is no EDU in a specific part of the template
        try:
            feature[:,:bert_dim] = self.stack[-1].embeddings
            feature[:,bert_dim:2*bert_dim] = self.stack[-2].embeddings
        except IndexError:
            pass
        try:
            feature[:,2*bert_dim:3*bert_dim] = self.queue[0].embeddings
            feature[:,3*bert_dim:4*bert_dim] = self.queue[1].embeddings
        except IndexError:
            pass
        try:
            feature[:,4*bert_dim:5*bert_dim] = self.paragraph[self.heads[self.stack[-1].id]-1].embeddings
        except (IndexError, KeyError):
            pass
        try:
            feature[:,5*bert_dim:6*bert_dim] = self.paragraph[self.heads[self.stack[-2].id]-1].embeddings
        except (IndexError, KeyError):
            pass
        return feature
        
    def shift(self):
        """
        push the top element of the queue to the stack
        """
        self.stack.append(self.queue.pop(0))
        return
        
    #set the header of the top element of the stack to be the element of the queue
    #and reduce
    def left_arc(self):
        """
        set the header of the top element of the stack to be the element of the queue
        and reduce
        """
        self.heads[self.stack[-1].id] = self.queue[0].id
        self.reduce()
        return
    
    def right_arc(self):
        """
        set the header of the top element of the queue to be the top element of the stack
        and shift
        """
        self.heads[self.queue[0].id] = self.stack[-1].id
        self.shift()
        return
        
    def reduce(self):
        """
        pop the top element of the queue
        """
        self.stack.pop()
        return
    
    def random_decide(self):
        """
        randomly decide a legal action
        """
        #return finish(-1) if none is left in queue
        if len(self.queue) == 0:
            return -1
        #return shift(0) is there is nothing in the stack
        if len(self.stack) == 0:
            return 0
        #true if the top element of queue has a head
        queue_head = self.queue[0].id in self.heads
        #true if the top element of stack has a head
        stack_head = self.stack[-1].id in self.heads
        
        #if both stack and queue not in headers
        if (not queue_head) and (not stack_head):
            return random.choice([0,1,2])
        
        #if only stack has a head
        if (not queue_head) and stack_head:
            return random.choice([0,2,3])
        
        #if only queue has a head
        if queue_head and (not stack_head):
            return random.choice([0,1])
        #if both have a head
        return random.choice([0,3])

    def possible_actions(self):
        """
        return possible actions in a certain situation
        """
        #return finish(-1) if none is left in queue
        if len(self.queue) == 0:
            return [-1]
        #return shift(0) is there is nothing in the stack
        if len(self.stack) == 0:
            return [0]
        #true if the top element of queue has a head
        queue_head = self.queue[0].id in self.heads
        #true if the top element of stack has a head
        stack_head = self.stack[-1].id in self.heads
        
        #if both stack and queue not in headers
        if (not queue_head) and (not stack_head):
            return [0,1,2]
        
        #if only stack has a head
        if (not queue_head) and stack_head:
            return [0,2,3]
        
        #if only queue has a head
        if queue_head and (not stack_head):
            return [0,1]
        #if both have a head
        return [0,3]
    
    def move(self,action):
        """
        move according to a given action
        0: shift
        1: left_arc
        2: right_arc
        3: reduce
        """
        if action == 0:
            self.shift()
        elif action == 1:
            self.left_arc()
        elif action == 2:
            self.right_arc()
        elif action == 3:
            self.reduce()
        return

    def golden_decide(self):
        """
        decide the golden action for the current state
        """
        #return finish(-1) if none is left in queue
        if len(self.queue) == 0:
            return -1
        #return shift(0) is there is nothing in the stack
        if len(self.stack) == 0:
            return 0
        #return leftarc(1) if the head of top element of stack is 
        #top of the queue
        if self.stack[-1].head == self.queue[0].id:
            return 1
        
        #return rightarc(2) if the head of top element of queue is
        #top of the stack
        if self.queue[0].head == self.stack[-1].id:
            return 2
        
        #if the head of the stack is greater than the queue, return 0
        if self.stack[-1].head > self.queue[0].id:
            return 0
        
        #if this is the root, perform shift
        if self.stack[-1].head == 0:
            return 0
        
        to_be_used = False
        for edu in self.queue:
            if edu.head == self.stack[-1].id:
                to_be_used = True
                break
        #perform reduce or shift if the stack has already got a head
        if self.stack[-1].id in self.heads:
            #return 0 if it is still to be used
            if to_be_used:
                return 0
            return 3
        
        #return 4 if something is wrong
        return 1
    
    def golden_execute(self,counter= None):
        """
        execuet the transition system with gold oracle
        """
        action = self.golden_decide()
        if counter != None:
                counter[action] += 1
        while(action!=-1):
            if action == 4:
                print("Aho something went wrong")
                break
            self.move(action)
            if counter != None:
                counter[action] += 1
            action = self.golden_decide()
        while(len(self.stack)!=0):
        #the bottom element of the stack must be the root
            edu = self.stack.pop()
            if not edu.id in self.heads:
                self.heads[edu.id] = 0
        return self.heads
    
    def model_execute(self,model,toprint = False, counter = None):
        """
        execute with a model given, np.argmax(model(x)) should give the action number
        if toprint: print out the action sequence
        if counter is passed: update the counter with the number of each action
        """
        feature = self.get_feature()
        action = np.argmax(model(feature))
        if counter != None:
                counter[action] += 1
        if toprint:
            print("The action is: "+str(action))
        while(len(self.queue)!=0):
            #if the predicted action is not allowed
            try:
                self.move(action)
            except:
                self.move(self.random_decide())
            feature = self.get_feature()
            action = np.argmax(model(feature))
            if counter != None and len(self.queue)!=0:
                counter[action] += 1
            if toprint:
                print("The action is: "+str(action))
        while(len(self.stack)!=0):
            edu = self.stack.pop()
            if not edu.id in self.heads:
                self.heads[edu.id] = 0
        return self.heads
    
    def random_execute(self, teacher_alpha = 0):
        """
        execute randomly
        with teacher_alpha probability to choose the gold action
        """
        action = self.random_decide()
        while(action!=-1):
            if action == 0:
                self.shift()
            elif action == 1:
                self.left_arc()
            elif action == 2:
                self.right_arc()
            else:
                self.reduce()
            if random.random() <= teacher_alpha:
              action = self.golden_decide()
            else:
              action = self.random_decide()
        while(len(self.stack)!=0):
            edu = self.stack.pop()
            if not edu.id in self.heads:
                self.heads[edu.id] = 0
        return self.heads
        
class Arc_eager(Transition_system):
    """
    the super class Transition_system does not support the case when 
    edu.id does not match its position in the discourse
    this class deal with this problem
    """
    def __init__(self, EDUs):
        """
        initialize the transition system
        with default configuration
        """
        #paralen is the length of the paragraph
        para_len = len(EDUs)
        self.para_len = para_len
        new_EDUs = []
        self.recovery_dict = {}
        self.recovery_dict[0] = 0
        self.construction_dict = {}
        for i in range(para_len):
            self.construction_dict[EDUs[i].id] = i+1
        for i in range(para_len):
            embeddings_copy = EDUs[i].embeddings
            EDUs[i].embeddings = 0
            new_EDU = copy.deepcopy(EDUs[i])
            EDUs[i].embeddings = embeddings_copy
            new_EDU.embeddings = embeddings_copy
            self.recovery_dict[i+1] = new_EDU.id
            new_EDU.id = i + 1
            #if the head is not in the current sentence, set it to 0
            try:
                new_EDU.head = self.construction_dict[new_EDU.head]
            except KeyError:
                new_EDU.head = 0
            new_EDUs.append(new_EDU)
        #initialize the transition system with new EDUs
        super().__init__(new_EDUs)
        return 
    
    def head_lookup(self):
        """
        recover heads
        """
        new_heads = {}
        for key, value in self.heads.items():
            new_heads[self.recovery_dict[key]] = self.recovery_dict[value]
        return new_heads
    
    #cover the original golden execute
    def golden_execute(self, counter = None):
        super().golden_execute(counter)
        return self.head_lookup()

    #use the parent execute but reverse the heads
    def model_execute(self,model,toprint = False, counter = None):
        super().model_execute(model, toprint, counter)
        return self.head_lookup()

    #use the parent random execute but reverse the heads
    def random_execute(self, teacher_alpha = 0):
        super().random_execute(teacher_alpha)
        return self.head_lookup()