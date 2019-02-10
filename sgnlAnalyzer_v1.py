#!/usr/bin/python

import Tkinter, Tkconstants, tkFileDialog
import os.path
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from numpy import diff
import cv2
from scipy import stats
from math import factorial
import pandas

class TkAPAnalyzerApp(Tkinter.Tk):
    
    def __init__(self, parent):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.filename = None
        self.resizable(True,False)
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        ch = 595
        self.geometry("%dx%d+0+0" % (0.99*w, ch))
##        self.geometry('1275x360')
        self.GUIbuilder()
    

    def GUIbuilder(self):

        FrameUno = Tkinter.Frame(self, bd=3, relief='solid')
        FrameUno.grid(column=0,columnspan=6,row=0,sticky='EW')

        Tkinter.Label(FrameUno,text="SgnlAnalyzer: Peak Finder and Analyzer", font=('Helvetica Neue',13,'bold'),
                      anchor="c",fg="white",bg="blue").grid(column=0,columnspan=6,row=0,sticky= 'EW',pady=10,padx=10)

        browseBttn = Tkinter.Button(FrameUno, text='Browser')
        browseBttn.grid(column=0,columnspan=1,row=1,pady=10,padx=20)
        browseBttn.configure(command = self.runBrowser)

        self.fileName = Tkinter.StringVar()
        self.fileName.set("Please Select File")
        fileLabel = Tkinter.Label(FrameUno, textvariable=self.fileName)
        fileLabel.grid(column=1,columnspan=3,row=1,sticky="W",pady=10,padx=20)

        Tkinter.Label(FrameUno,text="Acquisiton Speed (kHz):",anchor="c",fg="white",bg="black").grid(column=4,columnspan=1,row=1,sticky='EW',padx=10,pady=10)
        self.iAcqSpeed = Tkinter.DoubleVar()
        self.iAcqSpeed.set(0.02)
        self.iAcqSpeedEntry = Tkinter.Entry(FrameUno,textvariable=self.iAcqSpeed,width=30)
        self.iAcqSpeedEntry.grid(column=5,columnspan=1,row=1,sticky='EW',padx=10)  

        Tkinter.Label(FrameUno,text="Filtering Option:",anchor="c",fg="white",bg="black").grid(column=0,columnspan=1,row=2,sticky='EW',padx=10,pady=10)
        self.spinFreq = Tkinter.StringVar()
        self.fSBox = Tkinter.Spinbox(FrameUno, values=('No Filter','Opt1 (7)','Opt2 (31)','Opt3 (91)','Opt4 (151)','Opt5 (551)'),textvariable=self.spinFreq)
        self.spinFreq.set('No Filter')
        self.fSBox.grid(column=1,columnspan=1,row=2, sticky='EW',padx=25)

        self.checkCustomFlt = Tkinter.IntVar()
        self.checkCustomFlt.set(0)
        self.fltCB = Tkinter.Checkbutton(FrameUno, text="Custom Filtering", bg="gray", anchor ="c",variable=self.checkCustomFlt)
        self.fltCB.grid(column=2,columnspan=1,row=2, sticky='EW',padx=10,pady=10)

        self.customFlt = Tkinter.DoubleVar()
        self.customFltEntry = Tkinter.Entry(FrameUno,textvariable=self.customFlt,width=30)
        self.customFltEntry.grid(column=3,columnspan=1,row=2,sticky='EW',padx=10)

        Tkinter.Label(FrameUno,text="Offset (mV):",anchor="c",fg="white",bg="black").grid(column=4,row=2,sticky='EW',padx=10,pady=10)
        self.iOffset = Tkinter.DoubleVar()
        self.iOffsetEntry = Tkinter.Entry(FrameUno,textvariable=self.iOffset,width=30)
        self.iOffsetEntry.grid(column=5,columnspan=1,row=2,sticky='EW',padx=10)       

        Tkinter.Label(FrameUno,text="Define Display ROI?",anchor="c",fg="white",bg="black").grid(column=0,row=3,sticky='EW',padx=10,pady=10)

        self.checkROI = Tkinter.IntVar()
        self.checkROI.set(0)
        self.roiCB = Tkinter.Checkbutton(FrameUno, text="Yes (Check)", anchor ="w",variable=self.checkROI)
        self.roiCB.grid(column=1,row=3, sticky='EW')
        
        Tkinter.Label(FrameUno,text="From:",anchor="c",fg="white",bg="gray45").grid(column=2,columnspan=1,row=3,sticky= 'EW',pady=5,padx=10)
        self.iLimit = Tkinter.DoubleVar()
        self.iLimitEntry = Tkinter.Entry(FrameUno,textvariable=self.iLimit,width=30)
        self.iLimitEntry.grid(column=3,columnspan=1,row=3,sticky='EW',padx=10)

        Tkinter.Label(FrameUno,text="To:",anchor="c",fg="white",bg="gray45").grid(column=4,columnspan=1,row=3,sticky= 'EW',pady=5,padx=10)
        self.sLimit = Tkinter.DoubleVar()
        self.sLimitEntry = Tkinter.Entry(FrameUno,textvariable=self.sLimit,width=30)
        self.sLimitEntry.grid(column=5,columnspan=1,row=3,sticky='EW',padx=10)

        Tkinter.Label(FrameUno,text="Color AP Phases?",anchor="c",fg="white",bg="black").grid(column=0,row=4,sticky='EW',padx=10,pady=10)
        self.checkDrawArea = Tkinter.IntVar()
        self.checkDrawArea.set(1)
        self.drawAreaCB = Tkinter.Checkbutton(FrameUno, text="Yes (Check)", anchor ="w",variable=self.checkDrawArea)
        self.drawAreaCB.grid(column=1,row=4, sticky='EW')

        Tkinter.Label(FrameUno,text="Draw Slow-Depolarization Line?",anchor="c",fg="white",bg="black").grid(column=2,row=4,sticky='EW',padx=10,pady=10)
        self.checkDrawLine = Tkinter.IntVar()
        self.checkDrawLine.set(1)
        self.drawLineCB = Tkinter.Checkbutton(FrameUno, text="Yes (Check)", anchor ="w",variable=self.checkDrawLine)
        self.drawLineCB.grid(column=3,row=4, sticky='EW')

        Tkinter.Label(FrameUno,text="Draw Average Threshold Line?",anchor="c",fg="white",bg="black").grid(column=4,row=4,sticky='EW',padx=10,pady=10)
        self.checkDrawThresh = Tkinter.IntVar()
        self.checkDrawThresh.set(1)
        self.drawThreshCB = Tkinter.Checkbutton(FrameUno, text="Yes (Check)", anchor ="w",variable=self.checkDrawThresh)
        self.drawThreshCB.grid(column=5,row=4, sticky='EW')

        self.runButton = Tkinter.Button(FrameUno, state="disabled", text ="RUN",fg='red',font=('Helvetica Neue',16,'bold'))
        self.runButton.grid(column=2,columnspan=2,row=5,pady=30,padx=20)
        self.runButton.configure(command = self.runAnalysis)

        self.warningLabel = Tkinter.StringVar()
        self.warningLabel.set("")
        fileLabel = Tkinter.Label(FrameUno, textvariable=self.warningLabel,font=('Helvetica Neue',10,'bold'),anchor="c", fg="red")
        fileLabel.grid(column=0,columnspan=6,row=6,sticky="EW",pady=5,padx=10)

        Tkinter.Label(FrameUno,text="Analysis Summary Results", font=('Helvetica Neue',12,'bold'),
            anchor="c",fg="blue",bg="gray").grid(column=0,columnspan=6,row=7,sticky= 'EW',pady=10,padx=10)

        Tkinter.Label(FrameUno,text="Mean Frequency = ",anchor="c",fg="white",bg="gray45",font=('Helvetica Neue',9,'bold')).grid(column=1,columnspan=2,row=8,sticky= 'EW',pady=5,padx=10)
        self.avgFreq = Tkinter.StringVar()
        Tkinter.Label(FrameUno,textvariable=self.avgFreq,anchor="c",font=('Helvetica Neue',9,'bold')).grid(column=3,columnspan=1,row=8,sticky= 'W',pady=5,padx=10)

        Tkinter.Label(FrameUno,text="Mean Minimum Voltage = ",anchor="c",fg="white",bg="gray45",font=('Helvetica Neue',9,'bold')).grid(column=1,columnspan=2,row=9,sticky= 'EW',pady=5,padx=10)
        self.mnVoltage = Tkinter.StringVar()
        Tkinter.Label(FrameUno,textvariable=self.mnVoltage,anchor="c",font=('Helvetica Neue',9,'bold')).grid(column=3,columnspan=1,row=9,sticky= 'W',pady=5,padx=10)

        Tkinter.Label(FrameUno,text="Mean Threshold Voltage = ",anchor="c",fg="white",bg="gray45",font=('Helvetica Neue',9,'bold')).grid(column=1,columnspan=2,row=10,sticky= 'EW',pady=5,padx=10)
        self.thVoltage = Tkinter.StringVar()
        Tkinter.Label(FrameUno,textvariable=self.thVoltage,anchor="c",font=('Helvetica Neue',9,'bold')).grid(column=3,columnspan=1,row=10,sticky= 'W',pady=5,padx=10)
        
        Tkinter.Label(FrameUno,text="Mean Peak Voltage = ",anchor="c",fg="white",bg="gray45",font=('Helvetica Neue',9,'bold')).grid(column=1,columnspan=2,row=11,sticky= 'EW',pady=5,padx=10)
        self.mxVoltage = Tkinter.StringVar()
        Tkinter.Label(FrameUno,textvariable=self.mxVoltage,anchor="c",font=('Helvetica Neue',9,'bold')).grid(column=3,columnspan=1,row=11,sticky= 'W',pady=5,padx=10)

        Tkinter.Label(FrameUno,text="Mean Slow-Depolarization Voltage Rate = ",anchor="c",fg="white",bg="gray45",font=('Helvetica Neue',9,'bold')).grid(column=1,columnspan=2,row=12,sticky= 'EW',pady=5,padx=10)
        self.voltRate= Tkinter.StringVar()
        Tkinter.Label(FrameUno,textvariable=self.voltRate,anchor="c",font=('Helvetica Neue',9,'bold')).grid(column=3,columnspan=1,row=12,sticky= 'W',pady=5,padx=10)

        Tkinter.Label(FrameUno,text="Mean Fast-Depolarization Exponential Constant = ",anchor="c",fg="white",bg="gray45",font=('Helvetica Neue',9,'bold')).grid(column=1,columnspan=2,row=13,sticky= 'EW',pady=5,padx=10)
        self.voltFRate= Tkinter.StringVar()
        Tkinter.Label(FrameUno,textvariable=self.voltFRate,anchor="c",font=('Helvetica Neue',9,'bold')).grid(column=3,columnspan=1,row=13,sticky= 'W',pady=5,padx=10)

        FrameUno.columnconfigure(0,weight=1,uniform='a')
        FrameUno.columnconfigure(1,weight=1,uniform='a')
        FrameUno.columnconfigure(2,weight=1,uniform='a')
        FrameUno.columnconfigure(3,weight=1,uniform='a')
        FrameUno.columnconfigure(4,weight=1,uniform='a')
        FrameUno.columnconfigure(5,weight=1,uniform='a')
        
        self.update()
        self.geometry(self.geometry())
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=1)
        FrameUno.columnconfigure(0,weight=1)

            
    def runBrowser(self):

        self.file_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('Text Files', '.txt')]
##        options['initialdir'] = 'C:\\'
        options['title'] = 'Select File Containing AP Data (Time and Voltage)'

        self.fileName.set(tkFileDialog.askopenfilename(**self.file_opt))    

        if(os.path.isfile(self.fileName.get())):
            self.runButton.configure(state = "normal")

    def runAnalysis(self):

        fiLimit = self.iLimit.get()
        fsLimit = self.sLimit.get()

        intCheckOpt = 0
        if self.checkROI.get():
            intCheckOpt = 1

        if(os.path.isfile(self.fileName.get())):
            self.mainToolFunction(self.fileName.get(),intCheckOpt,fiLimit,fsLimit)  


    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
    ##    import numpy as np
    ##    from math import factorial
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
       
        return np.convolve( m[::-1], y, mode='valid')


    # =======================================================================================#


    # FOURIER DOMAIN: AP PERIOD CALCULATION =================================================#

    def peakFinder(self,Time, myVoltage, dt, myThresh):

        if fOpt == 'No Filter':
            smVolts = myVoltage
        else:
            smVolts = self.savitzky_golay(myVoltage, fCoeff, fOrder)  
       
        vMax = np.amax(smVolts)
        vMin = np.amin(smVolts)

        numPts = smVolts.shape[0]

        thrVoltage = stats.threshold(smVolts, threshmin=vMin, threshmax=myThresh, newval=255)
        thrVoltage = stats.threshold(thrVoltage, threshmin=255, threshmax=255, newval=0)
        diffThrV = np.diff(thrVoltage)

        myEdges = np.nonzero(diffThrV)
        myEdges = np.transpose(myEdges)

        npEdges = np.asarray(myEdges)
        numEdges = npEdges.shape[0]
        nPeaks = int(numEdges/2)
        timeSpacer = int(200/(dt*1e3))

        edgeFlag = 0

        if (numEdges % 2 == 0): #Even Number
            for n in range(0,nPeaks):
                tmpEdgeDiff=npEdges[(2*n)+1]-npEdges[2*n]
                if tmpEdgeDiff < timeSpacer:
                    edgeFlag = 1
                    self.warningLabel.set("WARNING: APs could not be properly identified. Try clicking at a different AP height or increase filtering setting.")
                else:
                    pass
        else: #Odd Number
            self.warningLabel.set("WARNING: APs could not be properly identified. Try clicking at a different AP height or increase filtering setting.")
            edgeFlag = 1

        myPeaks = np.zeros(nPeaks,dtype=float)
        myPeakIndexes = np.zeros(nPeaks,dtype=int)

        myMins = np.zeros(nPeaks+1,dtype=float)
        myMinIndexes = np.zeros(nPeaks+1,dtype=int)

        for p in range(0,nPeaks):
            myPeaks[p] = np.amax(myVoltage[int(npEdges[2*p]):int(npEdges[(2*p)+1])])
            myPeakIndexes[p] = int(npEdges[2*p])+np.argmax(myVoltage[int(npEdges[2*p]):int(npEdges[(2*p)+1])])

        if fCoeff == 0:
            myMins[0] = np.amin(myVoltage[0:int(myPeakIndexes[0])])
            myMinIndexes[0] = int(np.argmin(myVoltage[0:int(myPeakIndexes[0])]))
        else:
            myMins[0] = np.amin(self.savitzky_golay(myVoltage[0:int(myPeakIndexes[0])],fCoeff,fOrder))
            myMinIndexes[0] = int(np.argmin(self.savitzky_golay(myVoltage[0:int(myPeakIndexes[0])],fCoeff,fOrder)))
        
        for m in range(0,nPeaks-1):
            if fCoeff == 0:
                myMins[m+1] = np.amin(myVoltage[int(myPeakIndexes[m]):int(myPeakIndexes[m+1])])
                myMinIndexes[m+1] = int(myPeakIndexes[m]+np.argmin(myVoltage[int(myPeakIndexes[m]):int(myPeakIndexes[m+1])]))
            else:
                myMins[m+1] = np.amin(self.savitzky_golay(myVoltage[int(myPeakIndexes[m]):int(myPeakIndexes[m+1])],fCoeff,fOrder))
                myMinIndexes[m+1] = int(myPeakIndexes[m]+np.argmin(self.savitzky_golay(myVoltage[int(myPeakIndexes[m]):int(myPeakIndexes[m+1])],fCoeff,fOrder))) 

        if fCoeff == 0:
            myMins[-1] = np.amin(myVoltage[int(myPeakIndexes[-1]):numPts])
            myMinIndexes[-1] = int(myPeakIndexes[-1]+np.argmin(myVoltage[int(myPeakIndexes[-1]):numPts]))
        else:
            myMins[-1] = np.amin(self.savitzky_golay(myVoltage[int(myPeakIndexes[-1]):numPts],fCoeff,fOrder))
            myMinIndexes[-1] = int(myPeakIndexes[-1]+np.argmin(self.savitzky_golay(myVoltage[int(myPeakIndexes[-1]):numPts],fCoeff,fOrder)))

        myUnoBaseIndexes = np.zeros(nPeaks,dtype=int)
        myDosBaseIndexes = np.zeros(nPeaks,dtype=int)

        myUnoBaseIndexes[0] = np.argmax(np.diff(np.diff(myVoltage[0:int(myPeakIndexes[0])])/dt)/dt)-1

        for p in range(0,nPeaks-1):
            myUnoBaseIndexes[p+1] = np.argmax(np.diff(np.diff(myVoltage[int(myPeakIndexes[p]):int(myPeakIndexes[p+1])])/dt)/dt)
            myUnoBaseIndexes[p+1] = myUnoBaseIndexes[p+1]+myPeakIndexes[p]-1

        myMinDeriv = np.zeros(nPeaks,dtype=int)

        fltDVoltage = self.savitzky_golay(np.diff(myVoltage)/dt,7,2)

        for p in range(0,nPeaks-1):
            myMinDeriv[p] = np.argmin(fltDVoltage[myPeakIndexes[p]:myPeakIndexes[p+1]])
            myMinDeriv[p] = myMinDeriv[p]+myPeakIndexes[p]

        myMinDeriv[-1] = np.argmin(fltDVoltage[myPeakIndexes[-1]:fltDVoltage.shape[0]])+myPeakIndexes[-1]

        for p in range(0,nPeaks-1):
            slicedDVoltage = fltDVoltage[myMinDeriv[p]:myUnoBaseIndexes[p+1]]
            cont = 0
            while slicedDVoltage[cont]<0:
                cont = cont+1
            myDosBaseIndexes[p] = myMinDeriv[p]+cont
            del slicedDVoltage

        slicedDVoltage = fltDVoltage[myMinDeriv[-1]:fltDVoltage.shape[0]]
        cont = 0 
        while slicedDVoltage[cont]<0:
            cont = cont+1
        myDosBaseIndexes[-1] = myMinDeriv[-1]+cont
        del slicedDVoltage

##        fTempp = plt.figure(figsize=(16,8))
##        plt.plot(Time[:-2],self.savitzky_golay(np.diff(np.diff(myVoltage)/dt)/dt,11,2),'k-')
##        plt.plot(Time[:-1],self.savitzky_golay(np.diff(myVoltage)/dt,7,2),'k-')
             
        return nPeaks, myPeaks, myPeakIndexes, myMins, myMinIndexes, edgeFlag, myUnoBaseIndexes, myDosBaseIndexes


    # =======================================================================================#

    def onclick(self, event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        coords = np.array((ix,iy))
       
        if len(coords) == 2:

            fZero.canvas.mpl_disconnect(cid)
            plt.close(fZero)

            (nPeaks, myPeaks, myPeakIndexes, myMins, myMinIndexes, edgeFlag, myUnoBaseIndexes, myDosBaseIndexes) = self.peakFinder(Time, myVoltage, dt, iy)

            if edgeFlag == 0:

                # Period and Frequency Calculation ======================================================= #

                myAPPeriod = np.zeros(nPeaks-1,dtype=float)
                myFreqs = np.zeros(nPeaks-1,dtype=float)
                
                for m in range(0,nPeaks-1):
                    myAPPeriod[m] = Time[myPeakIndexes[m+1]]-Time[myPeakIndexes[m]]
                    myFreqs[m] = float((1/myAPPeriod[m])*60)

                avgFreq = np.mean(myFreqs)

                strAPFreq = '{:.2f}'.format(avgFreq)
                self.avgFreq.set(strAPFreq+'   APs/minute')

                lnPeak = np.amax(myPeaks)
                lnMins = np.amin(myMins)

                self.mnVoltage.set('{:.2f}'.format(np.mean(myMins[:-1]))+'   mV')
                self.mxVoltage.set('{:.2f}'.format(np.mean(myPeaks))+'   mV')

                AUno = np.zeros(nPeaks,dtype=float)
                tPeak = np.zeros(nPeaks,dtype=float)
                tEnd = np.zeros(nPeaks,dtype=float)
                tStart = np.zeros(nPeaks,dtype=float)
                sgnlMin = np.zeros(nPeaks,dtype=float)

                for k in range(0,nPeaks):
                    if myVoltage[myUnoBaseIndexes[k]]>myVoltage[myDosBaseIndexes[k]]:
                        sgnlMin[k] = myVoltage[myDosBaseIndexes[k]]
                        baseSignal = myVoltage[myUnoBaseIndexes[k]:myDosBaseIndexes[k]]-sgnlMin[k]
                    else:
                        sgnlMin[k] = myVoltage[myUnoBaseIndexes[k]]
                        baseSignal = myVoltage[myUnoBaseIndexes[k]:myDosBaseIndexes[k]]-sgnlMin[k]
                    AUno[k] = np.sum(baseSignal)
                    print AUno[k], dt, AUno[k]*dt
                    AUno[k] = AUno[k]*dt
                    tStart[k] = Time[myUnoBaseIndexes[k]]
                    tPeak[k] = Time[myPeakIndexes[k]]-Time[myUnoBaseIndexes[k]]
                    tEnd[k] = Time[myDosBaseIndexes[k]]-Time[myUnoBaseIndexes[k]]
                    del baseSignal



                dataFile = inFile[:-4]+'_ResultsSummary.xlsx'

                myHeader = ['FileName','Flash_Peak','Flash_StartTime','Time_To_Peak','Flash_Duration','Area']
                myNotes = {'FileName':'','Flash_Peak':'','Flash_StartTime':'','Time_To_Peak':'Ref. to Flash_StartTime','Flash_Duration':'','Area':''}
                myUnits = {'FileName':'','Flash_Peak':'F/F0','Flash_StartTime':'s','Time_To_Peak':'s','Flash_Duration':'s','Area':'a.u.*second'}

                myDF = pandas.DataFrame(columns=myHeader)
                myDF = myDF.append(myNotes, ignore_index=True)
                myDF = myDF.append(myUnits, ignore_index=True)
                
                for k in range(0,nPeaks):
                    data = {'FileName':inFile[:-4].split("/")[-1],'Flash_Peak':myPeaks[k],'Flash_StartTime':tStart[k],'Time_To_Peak':tPeak[k],'Flash_Duration':tEnd[k],'Area':AUno[k]}
                    myDF = myDF.append(data, ignore_index=True)

                myWriter = pandas.ExcelWriter(dataFile, engine='xlsxwriter')
                myDF.to_excel(myWriter,index=False,sheet_name='Results')           
                myWorkbook = myWriter.book
                myWorksheet = myWriter.sheets['Results']
                myWorksheet.set_zoom(90)
                myWorksheet.set_column('A:A',70)
                myWorksheet.set_column('B:F',20)
##                myWorksheet.set_column('I:J',25)
##                myWorksheet.set_column('K:P',20)
    ##            myWorkbook.add_format({'align':'center','valign':'vcenter'})

                myHeaderDos = ['Parameter','Value','SEM']
                myDFF = pandas.DataFrame(columns=myHeaderDos)
##
                nPeaksLine = {'Parameter':'Number of Flashes','Value':nPeaks,'SEM':''}
                freqLine = {'Parameter':'Frequency (Flashes/Minute)','Value':avgFreq,'SEM':stats.sem(myFreqs)}
##                threshLine = {'Parameter':'Voltage Threshold (mV)','Value':avgThrsh,'SEM':stats.sem(myThreshVolts)}
##                minLine = {'Parameter':'Minimum Voltage (mV)','Value':np.mean(myMins[:-1]),'SEM':stats.sem(myMins[:-1])}
                peakLine = {'Parameter':'Flash Peak (F/F0)','Value':np.mean(myPeaks),'SEM':stats.sem(myPeaks)}
##                tThreshLine = {'Parameter':'Threshold Time (s)','Value':np.mean(tThresh),'SEM':stats.sem(tThresh)}
                tPeakLine = {'Parameter':'Time To Peak (s)','Value':np.mean(tPeak),'SEM':stats.sem(tPeak)}
                tEndAPLine = {'Parameter':'Flash Duration (s)','Value':np.mean(tEnd),'SEM':stats.sem(tEnd)}
                restLine = {'Parameter':'Mean Flash Area ((a.u.)*s)','Value':np.mean(AUno),'SEM':stats.sem(AUno)}
##                depoLine = {'Parameter':'Average Depolarization Area (mV*seconds)','Value':np.mean(ADos),'SEM':stats.sem(ADos)}
##                repoLine = {'Parameter':'Average Repolarization Area (mV*seconds)','Value':np.mean(ATres),'SEM':stats.sem(ATres)}
##                pRestLine = {'Parameter':'Average Percent Resting Area (%)','Value':np.mean(pAUno),'SEM':stats.sem(pAUno)}
##                pDepoLine = {'Parameter':'Average Percent Depolarization Area (%)','Value':np.mean(pADos),'SEM':stats.sem(pADos)}
##                pRepoLine = {'Parameter':'Average Percent Repolarization Area (%)','Value':np.mean(pATres),'SEM':stats.sem(pATres)}
##                vrateLine = {'Parameter':'Slow-Depolarization Voltage Rate (mV/s)','Value':np.mean(voltageRate),'SEM':stats.sem(voltageRate)}
##                expCLine = {'Parameter':'Fast-Depolarization Exponential Constant (ms)','Value':np.mean(voltageFRate),'SEM':stats.sem(voltageFRate)}
##               
                myDFF = myDFF.append(nPeaksLine, ignore_index=True)
                myDFF = myDFF.append(freqLine, ignore_index=True)
##                myDFF = myDFF.append(minLine, ignore_index=True)
##                myDFF = myDFF.append(threshLine, ignore_index=True)
                myDFF = myDFF.append(peakLine, ignore_index=True)
##                myDFF = myDFF.append(tThreshLine, ignore_index=True)
                myDFF = myDFF.append(tPeakLine, ignore_index=True)
                myDFF = myDFF.append(tEndAPLine, ignore_index=True)
                myDFF = myDFF.append(restLine, ignore_index=True)
##                myDFF = myDFF.append(depoLine, ignore_index=True)
##                myDFF = myDFF.append(repoLine, ignore_index=True)
##                myDFF = myDFF.append(pRestLine, ignore_index=True)
##                myDFF = myDFF.append(pDepoLine, ignore_index=True)
##                myDFF = myDFF.append(pRepoLine, ignore_index=True)
##                myDFF = myDFF.append(vrateLine, ignore_index=True)
##                myDFF = myDFF.append(expCLine, ignore_index=True)
##
                myDFF.to_excel(myWriter,index=False,sheet_name='Summary (Means)')           
                myWorkbook = myWriter.book
                myWorksheet = myWriter.sheets['Summary (Means)']
                myWorksheet.set_zoom(90)
                myWorksheet.set_column('A:A',45)
                myWorksheet.set_column('B:C',20)
    ##            myWorkbook.add_format({'align':'center','valign':'vcenter'})
##
##
                myHeaderTres = ['Parameter','Value']
                myDFFF = pandas.DataFrame(columns=myHeaderTres)
##
                softwareVer = {'Parameter':'Software Version:','Value':'sgnlAnalyzer_v1'}
                fileLine = {'Parameter':'File-Name','Value':inFile}
                iAcqSpeedLine = {'Parameter':'Data Acquisition Speed (kHz):','Value':self.iAcqSpeed.get()}
                spinFreqLine= {'Parameter':'Filtering:','Value':fOpt}
##                offsetLine = {'Parameter':'Entered Offset (mV)','Value':self.iOffset.get()}
##
                myDFFF = myDFFF.append(softwareVer, ignore_index=True)              
                myDFFF = myDFFF.append(fileLine, ignore_index=True)
                myDFFF = myDFFF.append(iAcqSpeedLine, ignore_index=True)
                myDFFF = myDFFF.append(spinFreqLine, ignore_index=True)
##                myDFFF = myDFFF.append(offsetLine, ignore_index=True)
##
                myDFFF.to_excel(myWriter,index=False,sheet_name='Analysis Info')           
                myWorkbook = myWriter.book
                myWorksheet = myWriter.sheets['Analysis Info']
                myWorksheet.set_zoom(90)
                myWorksheet.set_column('A:A',30)
                myWorksheet.set_column('B:B',120)
    ##            myWorkbook.add_format({'align':'center','valign':'vcenter'})
                
                myWriter.save()

                self.iOffset.set(0.0)


                fUno = plt.figure(figsize=(16,8))
                plt.plot(Time,myVoltage,'k-')
                plt.xlabel('Time (Seconds)')
                plt.ylabel('Voltage (mV)')
                plt.xlim((minTime,maxTime))


                if self.checkDrawArea.get():

                    for p in range(0,nPeaks):
                        if p % 2 == 0:
                            plt.fill_between(Time, sgnlMin[p], myVoltage, where=((Time>=Time[myUnoBaseIndexes[p]])&(Time<=Time[myDosBaseIndexes[p]])), facecolor='lightgreen', interpolate=True)
                        else:
                            plt.fill_between(Time, sgnlMin[p], myVoltage, where=((Time>=Time[myUnoBaseIndexes[p]])&(Time<=Time[myDosBaseIndexes[p]])), facecolor='lightblue', interpolate=True)


                plt.savefig(inFile[:-4]+".png", dpi=500)

                plt.show()

            else:
                self.avgFreq.set('')
                self.thVoltage.set('')
                self.mnVoltage.set('')
                self.mxVoltage.set('')
                self.voltRate.set('')
                self.voltFRate.set('')


    def mainToolFunction(self,myFile,checkROI,mnTime,mxTime): # MAIN LOOP AND FUNCTION CALLING ===================================#

        global inFile, Time, myVoltage, dt, minTime, maxTime

        inFile = myFile

        if checkROI == 1:
            minTime = mnTime
            maxTime = mxTime

        try:
            Time, myVoltage = loadtxt(inFile,unpack = True,skiprows=1)
        except ImportError:
            print "Error Loading File"

        self.warningLabel.set("")

        
        NumDataPts = int(Time.shape[0])
        del Time
        dt = 1./(self.iAcqSpeed.get()*1e3)
        Time = np.arange(0.,NumDataPts*dt,dt,dtype=float)
          
        myVoltage = myVoltage-self.iOffset.get()

        if checkROI == 0:
            minTime = np.amin(Time)
            maxTime = np.amax(Time)            

        global fZero, cid, fCoeff, fOrder, fOpt

        fOrder = 5

        fOpt = self.spinFreq.get()
        
        if fOpt == 'No Filter':
            fCoeff = 0
        if fOpt == 'Opt1 (7)':
            fCoeff = 7
        if fOpt == 'Opt2 (31)':
            fCoeff = 31
        if fOpt == 'Opt3 (91)':
            fCoeff = 91
        if fOpt == 'Opt4 (151)':
            fCoeff = 151
        if fOpt == 'Opt5 (551)':
            fCoeff = 551

        fZero = plt.figure(figsize=(16,8))

        cbCustomFlt = 0
        if self.checkCustomFlt.get():
            cbCustomFlt = 1

        if cbCustomFlt == 0:
            if fOpt == 'No Filter':
                plt.plot(Time, myVoltage, 'k-')
            else:
                plt.plot(Time, self.savitzky_golay(myVoltage, fCoeff, fOrder), 'k-')
        if cbCustomFlt == 1:
            fCoeff = int(self.customFlt.get())
            if (fCoeff % 2 == 0):
                fCoeff = fCoeff-1               
            else:
                pass
            fOpt = '('+str(fCoeff)+')'
            fOpt = 'Custom '+fOpt
            plt.plot(Time, self.savitzky_golay(myVoltage, fCoeff, fOrder), 'k-')
                           
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Voltage (mV)')
        
        if checkROI == 1:
            plt.xlim(minTime,maxTime)
        if checkROI == 0:
            plt.xlim(np.amin(Time), np.amin(Time)+((np.amax(Time)-np.amin(Time))*1))

        cid = fZero.canvas.mpl_connect('button_press_event', self.onclick)


        plt.show()



if __name__ == "__main__":
    app = TkAPAnalyzerApp(None)
    app.title('SgnlAnalyzer v1 by Jorge Castorena (Dr. Michael J. Davis Laboratory)')
    app.mainloop()





