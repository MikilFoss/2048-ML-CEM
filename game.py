import random



class Board:

    def __init__(self) -> None:
        #self.state = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.state = [[2,4,2,4],[4,2,4,2],[4,2,4,2],[2,4,2,0]]
        self.score = 0
        self.isover = False
        
    def printBoard(self): #TODO: pretty this up
        print("\n")
        print(self.state[0])
        print(self.state[1])
        print(self.state[2])
        print(self.state[3])
        print("\n")
    
    def spawn(self):
        isfour = random.randint(1,4) == 4
        indices = []
        for i in range(0,4):
            for j in range(0,4):
                if self.state[i][j] == 0:
                    indices.append([i,j])
        index = random.randint(0,len(indices)-1)
        if isfour:
            self.state[indices[index][0]][indices[index][1]] = 4
        else:
            self.state[indices[index][0]][indices[index][1]] = 2

    def getState(self):
        for i in range(0,4):
            for j in range(0,4):
                if(self.state[i][j]== 0):
                    self.isover = False
                    return

        for i in range(0,3):
            for j in range(0,3):
                if(self.state[i][j]== self.state[i + 1][j] or self.state[i][j]== self.state[i][j + 1]):
                    self.isover = False
                    return

        for j in range(0,3):
            if(self.state[3][j]== self.state[3][j + 1]):
                self.isover = False
                return
                
        for i in range(0,3):
            if(self.state[i][3]== self.state[i + 1][3]):
                self.isover = False
                return


        self.isover = True

    def compress(self):
        changed = False
        new_board = []
        for i in range(0,4):
            new_board.append([0] * 4)
            
        for i in range(0,4):
            pos = 0
            for j in range(0,4):
                if(self.state[i][j] != 0):
                    new_board[i][pos] = self.state[i][j]
                    if(j != pos):
                        changed = True
                    pos += 1
        self.state = new_board
        return changed
    
    def merge(self):
        changed = False
        state = list(self.state)
        for i in range(0, 4):
            for j in range(0,3):
                if(state[i][j] == state[i][j + 1] and state[i][j] != 0):
                    self.score += state[i][j]
                    state[i][j] = state[i][j] * 2
                    state[i][j + 1] = 0
                    changed = True
        self.state = state    
        return changed

    def transpose(self):
        #reverse the matrix
        temp = []
        for i in range(0,4):
            temp.append([self.state[0][i],self.state[1][i],self.state[2][i],self.state[3][i]])
        self.state = temp

    def reverse(self):
        temp = []
        for row in self.state:
            row.reverse()
            temp.append(row)
        self.state = temp

    def moveLeft(self):
        changed1 = self.compress()
        changed2 = self.merge()
        changed = changed1 or changed2
        self.compress()
        return changed

    def moveRight(self):
        self.reverse()
        changed = self.moveLeft()
        self.reverse()
        return changed
    
    def moveUp(self):
        self.transpose()
        changed = self.moveLeft()
        self.transpose()
        return changed

    def moveDown(self):
        self.transpose()
        changed = self.moveRight()
        self.transpose()
        return changed

    def findmoves(self):
        # test right
        self.priorstate = self.state
        self.moveRight(True)
        if self.state != self.priorstate:
            self.state = self.priorstate
            self.validmoves[3] = 1
        else:
            self.validmoves[3] = 0
        
        self.moveLeft(True)
        if self.state != self.priorstate:
            self.state = self.priorstate
            self.validmoves[2] = 1
        else:
            self.validmoves[2] = 0

        self.moveDown(True)
        if self.state != self.priorstate:
            self.state = self.priorstate
            self.validmoves[1] = 1
        else:
            self.validmoves[1] = 0

        self.moveUp(True)
        if self.state != self.priorstate:
            self.state = self.priorstate
            self.validmoves[0] = 1
        else:
            self.validmoves[0] = 0
        


    
        self.findmoves()
        

game = Board()
game.spawn()
game.printBoard()
while not game.isover:
    print("move")
    inp = input()

    if inp == "w":
        changed = game.moveUp()
        if changed:
            game.spawn()
    elif inp == "s":
        changed = game.moveDown()
        if changed:
            game.spawn()
    elif inp == "a":
        changed = game.moveLeft()
        if changed:
            game.spawn()
    elif inp == "d":
        changed = game.moveRight()
        if changed:
            game.spawn()
    elif inp == "admin":
        break
    else:
        print("nuh-uh")
    game.getState()
    game.printBoard()
print(game.score)
