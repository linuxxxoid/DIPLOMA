# импортирование модулей python
import tkinter as tk

# класс дочернего окна
class Dialog_window: #dialog
  def __init__(self, root):
    self.top = tk.Toplevel(root)
    self.top.title('dialog')
    self.top.geometry('400x100+300+250')
    self.frame = tk.Frame(self.top)
    self.frame.pack(side = tk.BOTTOM)
    self.accept_button = tk.Button(self.frame,
                                text = 'accept',
                                command = self.accept)
    self.accept_button.pack(side = tk.LEFT)


    self.cancel_button = tk.Button(self.frame,
                                text = 'cancel',
                                command = self.cancel)
    self.cancel_button.pack(side = tk.RIGHT)

    
    self.text = tk.Text(self.top,
                     background = 'white')
    self.text.pack(side = tk.TOP,
                   fill = tk.BOTH,
                   expand = tk.YES)
    self.top.protocol('WM_DELETE_WINDOW', self.cancel)

  def go(self, myText = '',):
    self.text.insert('0.0', myText)
    self.newValue = None
    self.top.grab_set()
    self.top.focus_set()
    self.top.wait_window()
    return self.newValue

  def accept(self):
    self.newValue = self.text.get('0.0', tk.END)
    self.top.destroy()

  def cancel(self):
    self.top.destroy()

  def vertical_hit(self):

    video_vertic_name = '/Users/linuxoid/Desktop/VUZICH/DIPLOMA/etalon videos/vertic.mp4'

    video_vertic = imageio.get_reader(video_vertic_name)

    my_label = tk.Label(self.root)
    my_label.pack(side="top", fill="both", expand=True)
    thread = threading.Thread(target=self.stream, args=(my_label, video_vertic))
    thread.daemon = 1
    thread.start()


  def stream(self, label, video):

    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        label.config(image=frame_image)
        label.image = frame_image



  def horizontal_hit(self):

    video_horiz_name = '/Users/linuxoid/Desktop/VUZICH/DIPLOMA/etalon videos/diag horiz.mp4'
    video_horiz = imageio.get_reader(video_horiz_name)
    my_label = tk.Label(self.root)
    my_label.pack(side="top", fill="both", expand=True)
    thread = threading.Thread(target=self.stream, args=(my_label, video_horiz))
    thread.daemon = 2
    thread.start()



  def horizontal_hit(self):

    video_horiz_name = '/Users/linuxoid/Desktop/VUZICH/DIPLOMA/etalon videos/diag horiz.mp4'
    video_horiz = imageio.get_reader(video_horiz_name)
    my_label = tk.Label(self.root)
    my_label.pack(side="top", fill="both", expand=True)
    thread = threading.Thread(target=self.stream, args=(my_label, video_horiz))
    thread.daemon = 2
    thread.start()


# тестовая команда
if __name__ == '__main__':
  root = tk.Tk()
  root.withdraw()
  myTest = dialog(root)
  print(myTest.go('Hello World!'))