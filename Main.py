import tkinter
from UI import gui_setup


if __name__ == "__main__":
    # call and name the UI
    root = tkinter.Tk()
    root.title("Car Price Estimation")

    # set up and the run the UI window from UI.py
    gui_setup(root)
    # set window size and run
    root.geometry('330x500')
    root.mainloop()
