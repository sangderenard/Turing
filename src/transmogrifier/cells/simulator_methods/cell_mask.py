def get_cell_mask(self, cell):
    return self.bitbuffer[cell.left:cell.right]

def set_cell_mask(self, cell, mask):
    self.bitbuffer[cell.left:cell.right] = mask

def pull_cell_mask(self, cell):
    cell._buf = self.get_cell_mask(cell)

def push_cell_mask(self, cell):
    self.set_cell_mask(cell, cell._buf)
