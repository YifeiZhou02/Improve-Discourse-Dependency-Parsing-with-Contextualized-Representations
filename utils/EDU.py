class EDU:
    """
    a data structure to represent an EDU
    """
    def __init__(self,EDU_content, embeddings):
        """
        initialize the data with EDU_content
        EDU_content is a list of edu_id, head id, relation, text
        sentenceNo, sentence ID
        """
        self.id= EDU_content[0]
        self.sentence = EDU_content[3]
        #calculate the dense representation for the EDU
        self.embeddings = None
        self.head = EDU_content[1]
        self.relation = EDU_content[2]
        self.sentenceNo = EDU_content[4]
        self.sentenceID = EDU_content[5]
    def __str__(self):
        return self.sentence
    
    def update_embeddings(self, embeddings):
        """
        update the embeddings of self(self.embeddings will be used as features)
        """
        self.embeddings = embeddings