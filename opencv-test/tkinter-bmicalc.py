from tkinter import *

root = Tk()
root.title("BMI Calculator")

title = Label(root, text="BMI CALCULATOR").grid(row=0, column=0, sticky='w')

askFt = Label(root, text="Feet", anchor='w').grid(row=2, column=0, sticky='w')
ft = Entry(root, width=10, borderwidth=10)
ft.grid(row=3, column=0, columnspan=1, padx=0, pady=10, sticky='w')

askIn = Label(root, text="Inches").grid(row=2, column=0)
inch = Entry(root, width=10, borderwidth=10)
inch.grid(row=3, column=0, columnspan=2, padx=0, pady=10)

askLb = Label(root, text="Pounds", anchor='w').grid(row=4, column=0, sticky='w')

lb = Entry(root, width=30, borderwidth=10)
lb.grid(row=5, column=0, columnspan=1, padx=10, pady=10, sticky='w')


def calculate():
    inchTotal = float(ft.get()) * 12 + float(inch.get())
    bmi = 703 * (float(lb.get()) / float(pow(inchTotal, 2)))

    bmiTell = Label(root, text= "Your BMI is:", anchor='w').grid(row=10, column=0, sticky='w')
    bmiDis = Label(root, text= bmi, anchor='w').grid(row=12, column=0, sticky='w')



calculate_button = Button(root, text="Calculate", command=calculate)
calculate_button.grid(row=6, column=0, pady=10)

root.mainloop()
