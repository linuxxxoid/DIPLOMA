# импортирование модулей python
import tkinter as tk

# класс диалогового окна выхода
class Exit_window:
  def __init__(self, root):
    self.slave = tk.Toplevel(root)
    self.frame = tk.Frame(self.slave)
    self.frame.pack(side = tk.BOTTOM)
    self.yes_button = tk.Button(self.frame, 
                             text = 'Yes', 
                             command = self.yes)
    self.yes_button.pack(side = tk.LEFT)
    self.no_button = tk.Button(self.frame, 
                            text = 'No', 
                            command = self.no) 
    self.no_button.pack(side = tk.RIGHT)   
    self.label = tk.Label(self.slave)
    self.label.pack(side = tk.TOP,
                    fill = tk.BOTH,
                    expand = tk.YES)
    self.slave.protocol('WM_DELETE_WINDOW', self.no)

  def go(self, title = 'question', 
               message = '[question goes here]', 
               geometry = '200x70+300+265'):
    self.slave.title(title)
    self.slave.geometry(geometry)
    self.label.configure(text = message)
    self.boolean_value = tk.TRUE
    self.slave.grab_set()
    self.slave.focus_set()
    self.slave.wait_window()
    return self.boolean_value

  def yes(self):
    self.boolean_value = tk.TRUE
    self.slave.destroy()

  def no(self):
    self.boolean_value = tk.FALSE
    self.slave.destroy()

# тестовая команда
if __name__ == '__main__':
  root = tk.Tk()
  root.withdraw()
  myTest = Exit_window(root)
  if myTest.go(message = 'Is it working?'):
    print('Yes')
  else:
    print('No')