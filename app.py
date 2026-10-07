import json
import tkinter as tk
from tkinter import ttk,messagebox
import numpy as np
from PIL import Image,ImageDraw,ImageTk
from model import Recognizer,prepare_drawing,ROOT

class DigitApp:
    def __init__(self,root):
        self.root=root;self.recognizer=Recognizer();self.samples=np.load(ROOT/'data'/'test_samples.npz');self.sample_index=None;self.last=None
        self.image=Image.new('L',(280,280),0)
        root.title('Handwritten Digit Recognition');root.geometry('700x650');root.minsize(600,580);root.configure(bg='#F4F2F8')
        root.columnconfigure(0,weight=1)
        header=tk.Frame(root,bg='#4C356A',padx=18,pady=16);header.grid(row=0,column=0,sticky='ew')
        tk.Label(header,text='Handwritten Digit Recognition',font=('Segoe UI',20,'bold'),bg='#4C356A',fg='white').pack(anchor='w')
        tk.Label(header,text='MNIST | Convolutional Neural Network | Digits 0-9',bg='#4C356A',fg='white').pack(anchor='w')
        tk.Label(root,text='Draw one large digit with the left mouse button, then click Predict.',bg='#F4F2F8').grid(row=1,column=0,pady=12)
        self.canvas=tk.Canvas(root,width=280,height=280,bg='black',highlightthickness=1)
        self.canvas.grid(row=2,column=0,pady=5)
        self.canvas.bind('<Button-1>',self.start_stroke);self.canvas.bind('<B1-Motion>',self.draw);self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'last',None))
        buttons=tk.Frame(root,bg='#F4F2F8');buttons.grid(row=3,column=0,pady=10)
        self.predict_button=ttk.Button(buttons,text='Predict',command=self.predict);self.predict_button.pack(side='left',padx=6)
        ttk.Button(buttons,text='Clear',command=self.clear).pack(side='left',padx=6)
        ttk.Button(buttons,text='Evaluation',command=self.evaluation).pack(side='left',padx=6)
        samples=tk.Frame(root,bg='#F4F2F8');samples.grid(row=4,column=0,pady=6)
        tk.Label(samples,text='Test sample:',bg='#F4F2F8').pack(side='left')
        self.selector=ttk.Combobox(samples,state='readonly',values=[f'Sample {i+1}' for i in range(200)],width=16);self.selector.pack(side='left',padx=6);self.selector.current(0)
        self.selector.bind('<<ComboboxSelected>>',lambda e:self.load_sample())
        ttk.Button(samples,text='Load sample',command=self.load_sample).pack(side='left')
        self.result=tk.StringVar(value='Draw a digit or load a test sample.')
        tk.Label(root,textvariable=self.result,bg='#F4F2F8',font=('Segoe UI',12),wraplength=650,justify='center').grid(row=5,column=0,padx=16,pady=15)
        tk.Label(root,text='Recognizes single digits only. Your drawings may differ from MNIST.',bg='#F4F2F8',fg='#655B70').grid(row=6,column=0,pady=8)
        root.bind('<Return>',self.predict)

    def start_stroke(self,event):
        if self.sample_index is not None:self.clear()
        self.sample_index=None;self.last=(max(0,min(279,event.x)),max(0,min(279,event.y)))
        x,y=self.last;ImageDraw.Draw(self.image).ellipse((x-7,y-7,x+7,y+7),fill=255)
        self.render()
    def draw(self,event):
        x,y=max(0,min(279,event.x)),max(0,min(279,event.y))
        if self.last is not None:
            draw=ImageDraw.Draw(self.image);draw.line((*self.last,x,y),fill=255,width=16);draw.ellipse((x-8,y-8,x+8,y+8),fill=255)
        self.last=(x,y);self.sample_index=None;self.render();self.result.set('Click Predict to recognize your drawing.')
    def render(self):
        self.preview=ImageTk.PhotoImage(self.image)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.preview,anchor='nw')
    def clear(self):
        self.image=Image.new('L',(280,280),0);self.sample_index=None;self.last=None
        self.canvas.delete('all');self.result.set('Draw a new digit or load a test sample.')
    def load_sample(self):
        index=self.selector.current()
        if index<0:return
        self.sample_index=index
        self.image=Image.fromarray(self.samples['images'][index]).resize((280,280),Image.Resampling.NEAREST)
        self.render();self.result.set(f'Sample {index+1} loaded. Click Predict.')
    def predict(self,event=None):
        try:
            pixels=self.samples['images'][self.sample_index] if self.sample_index is not None else prepare_drawing(self.image)
            digit,scores=self.recognizer.predict(pixels)
            top=np.argsort(scores)[::-1][:3]
            text=f'Predicted digit: {digit}\nTop scores: '+', '.join(f'{i}: {scores[i]:.1%}' for i in top)
            if self.sample_index is not None:
                actual=int(self.samples['labels'][self.sample_index]);text+=f'\nRecorded label: {actual} | '+('Match' if actual==digit else 'Mismatch')
            self.result.set(text)
        except ValueError as error:self.result.set(str(error))
        except Exception as error:
            print('Prediction error:',error);self.result.set('Prediction failed. Check the terminal for details.')
    def evaluation(self):
        m=json.loads((ROOT/'results'/'metrics.json').read_text())
        messagebox.showinfo('MNIST evaluation',f'Training: {m["training_samples"]}\nValidation: {m["validation_samples"]}\nTest: {m["test_samples"]}\n\nTest accuracy: {m["test_accuracy"]:.2%}\n\nModel selected using validation accuracy.\nTest data was not used for training.\nDrawing accuracy may be lower.',parent=self.root)

if __name__=='__main__':
    root=tk.Tk();DigitApp(root);root.mainloop()
