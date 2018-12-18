#!/bin/python3


########################################################
## Packaged
## 
########################################################
#


# IMPORT MODULES
########################################################

import os
import datetime
import re
import sys
import time
import numpy
import matplotlib.pyplot as plt
import hashlib
import random
import numpy as np
from shutil import copyfile

# CLASSES
########################################################



# CLASS: PWSCF_EOS
######################################

class pwscf_eos:

  def __init__(self):
    now = datetime.datetime.now()
    time_now = str(now.hour) + ":" + str(now.minute) + "   " + str(now.day) + "/" + str(now.month) + "/" + str(now.year)
    self.log("##########################################")
    self.log("               PWscf EOS                  ")
    self.log("##########################################")
    self.log("")
    self.log("Date + Time: " + time_now)
    self.log("")
    self.reset()
    try:
      self.set_template(sys.argv[1])
    except:
      self.set_template("")

  def reset(self):
    self.cwd = os.getcwd()
    fh = open("log.txt", 'w')
    fh.write("")
    fh.close()
    self.working_dir = "wd"
    self.data = {
    "relaxed_alat":  0.0,
    "density": 0.0,
    "isolated":  0.0,
    "alat":  0.0,
    "V0":  0.0,
    "E0":  0.0,
    "B0":  0.0,
    "B0P":  0.0,
    "C11":  0.0,
    "C22":  0.0,
    "C33":  0.0,
    "C44":  0.0,
    "C55":  0.0,
    "C66":  0.0,
    "C12":  0.0,
    "C13":  0.0,
    "C23":  0.0,
    "B0_gpa":  0.0,
    "C11_gpa":  0.0,
    "C22_gpa":  0.0,
    "C33_gpa":  0.0,
    "C44_gpa":  0.0,
    "C55_gpa":  0.0,
    "C66_gpa":  0.0,
    "C12_gpa":  0.0,
    "C13_gpa":  0.0,
    "C23_gpa":  0.0
    }
  # Run

  def run(self):
    self.start_time = time.time()
    self.verbose = True
    self.load_template()
    self.relax()
    self.eos()
    self.isolated()
    self.ec()
    #self.ec_test()
    self.output()

  def st(self):
    now = time.time()
    now = str((now - self.start_time)).strip()
    return now

  def load_template(self):
    if(self.verbose):
      print(self.st(), "Loading template")
    self.pwi = pwscf_input()
    self.pwi.load(self.template)
    self.pwi.set_dirs()
    self.original_file = self.pwi.get_data()
    # Load config - need to rewrite this
    self.pwi.load_config("FCC", 2, 15.2)

  def relax(self):
    if(self.verbose):
      print(self.st(), "Relax")
    self.log("Relax")
    dir = self.cwd + "/" + "wd/relax"
    self.pwi.set_calculation("vc-relax")
    self.pwi.save("relax.in", dir)
    pwexec = pwscf_exec("relax.in", dir)
    output = pwexec.run()
    # Read file
    pwo = pwscf_output(output[0])
    cp = pwo.get_cell_parameters()
    self.data['relaxed_alat'] = pwo.get_alat()
    self.data['density'] = pwo.get_density()
    if(self.verbose):
      print(self.st(), "Relax SCF Check")
    dir = self.cwd + "/" + "wd/scf"
    self.pwi.set_calculation("scf")
    self.pwi.set_cp(cp)
    self.pwi.nomalise_cell_parameters()
    self.pwi.save("relaxed.in", dir)
    pwexec = pwscf_exec("relaxed.in", dir)
    output = pwexec.run()
    # Read file
    pwo = pwscf_output(output[0])
    #pwo.output_details()

  def eos(self):
    if(self.verbose):
      print(self.st(), "EOS Calcs")
    self.pwi.set_calculation("scf")
    dir = self.cwd + "/" + "wd/eos"
    cp_arr = self.pwi.get_cp_array()
    # Set up list
    cp = [['alat'],['','',''],['','',''],['','','']]
    pwexec = pwscf_exec()
    size = 15
    strain = []
    nat = 0
    for i in range(size):
      s = numpy.zeros((3,3))
      strain.append(1.0 + (i - size // 2) * 0.001)
      for j in range(3):
        s[j,j] = 1.0 + (i - size // 2) * 0.001
      m = numpy.matmul(s, cp_arr)
      self.pwi.set_cp_arr(m)
      self.pwi.save("eos_" + str(i) + ".in", dir)
      nat = self.pwi.get_nat()
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    eos = self.process_results({"x": volume, "y": energy, "x_axis": "Volume", "y_axis": "Energy", "fit": "eos"}, None, "eos")
    self.data["eos_fit"] = eos
    self.data["V0"] = eos["V0"]
    self.data["E0"] = eos["E0"]
    self.data["B0"] = eos["B0"]
    self.data["B0P"] = eos["B0P"]
    self.data["B0_gpa"] = eos["B0"] * 14710.5
    self.data["alat"] = eos["V0"]**(1/3)
    self.data["E0_per_atom"] = eos["E0"] / nat

  def isolated(self):
    if(self.verbose):
      print(self.st(), "Isolated Atom")
    dir = self.cwd + "/" + "wd/isolated"
    pwiso = pwscf_input()
    pwiso.load_data(self.original_file)
    pwiso.set_as_isolated()
    alat = []
    pwexec = pwscf_exec()
    pwexec.set_terminate_at_error()
    for i in range(12):
      alat.append(6 + i)
      pwiso.set_alat(6 + i)
      pwiso.save("i_" + str(i) + ".in", dir)
      files = pwexec.add_file(pwiso.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    alat = alat[0:len(volume)]
    fit_alat = self.process_results({"x": alat, "y": energy, "x_axis": "Alat", "y_axis": "Energy", "fit": "isolated"}, None, "isolated_alat")
    e = pwscf_eos.isolated_fit(fit_alat, 20.0)
    self.data["e_isolated"] = e
    self.data["e_coh"] = self.data["E0_per_atom"] - self.data["e_isolated"]
    self.data["e_coh"] = 1.360570e1 * self.data["e_coh"]

  def ec_test(self):
    d = []
    delta = 0.001
    d.append([[1.0 + delta, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
    d.append([[1.0, 0.0, 0.0],[0.0, 1.0 + delta, 0.0],[0.0, 0.0, 1.0]])
    d.append([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0 + delta]])
    d.append([[1.0, 0.0, 0.0],[0.0, 1.0, delta],[0.0, delta, 1.0]])
    d.append([[1.0, delta, 0.0],[delta, 1.0, 0.0],[0.0, 0.0, 1.0]])
    d.append([[1.0, 0.0, delta],[0.0, 1.0, 0.0],[delta, 0.0, 1.0]])
    self.pwi.set_calculation("scf")
    relaxed_cp = self.pwi.get_cp_array
    dir = self.cwd + "/" + "wd/ec_test"
    pwexec = pwscf_exec()
    for i in range(len(d)):
      cp = self.pwi.set_cp_arr(d[i])
      self.pwi.save("strain_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()

  def ec(self):
    self.printl("Elastic Constants")
    self.pwi.set_calculation("scf")
    relaxed_cp = self.pwi.get_cp_array
    dir = self.cwd + "/" + "wd/ec"
    # C11
    #===============================
    if(self.verbose):
      print(self.st(), "D1  C11")
    pwexec = pwscf_exec()
    size = 15
    strain = []
    for i in range(size):
      d = 0.001 * (i - size // 4)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      cp = self.pwi.set_cp_arr([[1.0 + d,0,0],[0,1,0],[0,0,1]])
      self.pwi.save("c11_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "C11")
    self.data["C11_fit"] = c
    self.data["C11"] = (2 / self.data["V0"]) * c[0]
    self.data["C11_gpa"] = (2 / self.data["V0"]) * c[0] * 1.47105E4
    # C22
    #===============================
    if(self.verbose):
      print(self.st(), "D2  C22")
    pwexec = pwscf_exec()
    size = 15
    strain = []
    for i in range(size):
      d = 0.001 * (i - size // 4)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      cp = self.pwi.set_cp_arr([[1.0,0,0],[0,1.0 + d,0],[0,0,1]])
      self.pwi.save("c22_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "C22")
    self.data["C22_fit"] = c
    self.data["C22"] = (2 / self.data["V0"]) * c[0]
    self.data["C22_gpa"] = (2 / self.data["V0"]) * c[0] * 1.47105E4
    # C33
    #===============================
    if(self.verbose):
      print(self.st(), "D3  C33")
    pwexec = pwscf_exec()
    size = 15
    strain = []
    for i in range(size):
      d = 0.001 * (i - size // 4)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      cp = self.pwi.set_cp_arr([[1.0,0,0],[0,1.0,0],[0,0,1 + d]])
      self.pwi.save("c33_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "C33")
    self.data["C33_fit"] = c
    self.data["C33"] = (2 / self.data["V0"]) * c[0]
    self.data["C33_gpa"] = (2 / self.data["V0"]) * c[0] * 1.47105E4
    # C44
    #===============================
    if(self.verbose):
      print(self.st(), "D4  C44")
    pwexec = pwscf_exec()
    size = 11
    strain = []
    for i in range(size):
      d = 0.0005 * (i)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      u = ((1-d**2)**(1/3))
      cp = self.pwi.set_cp_arr([[1/u,0,0],[0,1/u,d/u],[0,d/u,1/u]])
      self.pwi.save("c44_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "C44")
    self.data["C44_fit"] = c
    self.data["C44"] = (1 / (2 * self.data["V0"])) * c[0]
    self.data["C44_gpa"] = self.data["C44"] * 1.47105E4
    # C55
    #===============================
    if(self.verbose):
      print(self.st(), "D5  C55")
    pwexec = pwscf_exec()
    size = 11
    strain = []
    for i in range(size):
      d = 0.0005 * (i)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      u = ((1-d**2)**(1/3))
      cp = self.pwi.set_cp_arr([[1/u,0,d/u],[0,1/u,0],[d/u,0,1/u]])
      self.pwi.save("c55_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "C55")
    self.data["C55_fit"] = c
    self.data["C55"] = (1 / (2 * self.data["V0"])) * c[0]
    self.data["C55_gpa"] = self.data["C55"] * 1.47105E4
    # C66
    #===============================
    if(self.verbose):
      print(self.st(), "D6  C66")
    pwexec = pwscf_exec()
    size = 11
    strain = []
    for i in range(size):
      d = 0.0005 * (i)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      u = ((1-d**2)**(1/3))
      cp = self.pwi.set_cp_arr([[1/u,d/u,0],[d/u,1/u,0],[0,0,1/u]])
      self.pwi.save("c66_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "C66")
    self.data["C66_fit"] = c
    self.data["C66"] = (1 / (2 * self.data["V0"])) * c[0]
    self.data["C66_gpa"] = self.data["C66"] * 1.47105E4
    # D7
    #===============================
    if(self.verbose):
      print(self.st(), "D7  C12, C13, C23")
    pwexec = pwscf_exec()
    size = 11
    strain = []
    for i in range(size):
      d = 0.0005 * (i)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      u = ((1-d**2)**(1/3))
      cp = self.pwi.set_cp_arr([[(1+d)/u,0,0],[0,(1-d)/u,0],[0,0,1/u]])
      self.pwi.save("D7_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "D7")
    self.data["D7_fit"] = c
    self.data["D7"] = c[0]
    # D8
    #===============================
    if(self.verbose):
      print(self.st(), "D8  C12, C13, C23")
    pwexec = pwscf_exec()
    size = 11
    strain = []
    for i in range(size):
      d = 0.0005 * (i)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      u = ((1-d**2)**(1/3))
      cp = self.pwi.set_cp_arr([[(1+d)/u,0,0],[0,1/u,0],[0,0,(1-d)/u]])
      self.pwi.save("D8_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "D8")
    self.data["D8_fit"] = c
    self.data["D8"] = c[0]
    # D9
    #===============================
    if(self.verbose):
      print(self.st(), "D9  C12, C13, C23")
    pwexec = pwscf_exec()
    size = 11
    strain = []
    for i in range(size):
      d = 0.0005 * (i)
      self.printl(str(i) + " " + str(d))
      strain.append(d)
      u = ((1-d**2)**(1/3))
      cp = self.pwi.set_cp_arr([[1/u,0,0],[0,(1+d)/u,0],[0,0,(1-d)/u]])
      self.pwi.save("D9_" + str(i) + ".in", dir)
      pwexec.add_file(self.pwi.get_path())
    output = pwexec.run()
    volume, energy = pwscf_eos.ve(output)
    c = self.process_results({"x": strain, "y": energy, "x_axis": "Strain", "y_axis": "Energy", "fit": "poly2"}, None, "D9")
    self.data["D9_fit"] = c
    self.data["D9"] = c[0]
    # C12, C13, C23
    #===============================
    self.data["C12"] = (self.data["C11"] + self.data["C22"]) / 2 - (self.data["D7"] / self.data["V0"])
    self.data["C13"] = (self.data["C11"] + self.data["C33"]) / 2 - (self.data["D8"] / self.data["V0"])
    self.data["C23"] = (self.data["C22"] + self.data["C33"]) / 2 - (self.data["D9"] / self.data["V0"])
    self.data["C12_gpa"] = self.data["C12"] * 1.47105E4
    self.data["C13_gpa"] = self.data["C13"] * 1.47105E4
    self.data["C23_gpa"] = self.data["C23"] * 1.47105E4
  #def process_results(self, output, dir=None, file_stub="", fit=None):

  def process_results(self, data_in, dir=None, file_stub=""):
  #
  #   Saves data file
  #   Creates plot and fits curve
  #
    # Make dirs
    if(dir == None):
      dir = self.make_dir(self.cwd + "/" + "wd")
    dir_data = self.make_dir(dir + "/" + "data")
    dir_plots = self.make_dir(dir + "/" + "plots")
    # Get data
    x = data_in['x']
    y = data_in['y']
    pwscf_eos.to_csv(dir_data+"/"+str(file_stub)+".csv", [x, y])
    # Fit type
    fit = data_in['fit']
    plt.clf()
    plt.title(file_stub)
    plt.ylabel("Energy")
    plt.xlabel("Volume")
    fit_c = [0]
    if(fit.lower() == "poly2"):
      fit_c = numpy.polyfit(x, y, 2)
      xfit, yfit = pwscf_eos.make_points(x, fit_c, "poly2")
      plt.plot(xfit, yfit, 'b-', label="E-V fit")
    elif(fit.lower() == "eos"):
      fit_c = fitting.eos_guess(x, y)
      xfit, yfit = pwscf_eos.make_points(x, fit_c, "eos")
      plt.plot(xfit, yfit, 'b-', label="E-V fit")
    elif(fit.lower() == "isolated"):
      data = [x,y]
      print(data)
      lma_fit = lma(data, True)
      p = [0,0,0,0]
      lma_fit.set_fit(pwscf_eos.isolated_fit, p)
      lma_fit.set_sa({"temp_start": 10.0, "temp_end": 0.01, "factor": 0.9, "count": 5000}, [-100,-10,-3,-20], [50,10,3,20])
      fit_c, rss = lma_fit.calc()
      xfit, yfit = pwscf_eos.make_points(x, fit_c, "isolated")
      plt.plot(xfit, yfit, 'b-', label="E-V fit")
    # Output plot
    plt.plot(x, y, 'r+', label="E-V")
    plt.savefig(dir_plots+'/'+str(file_stub)+'.eps', format='eps')
    return fit_c

  def output(self):
    print()
    print()
    print("Final Results")
    print("================================================")
    for key in sorted(self.data.keys()):
      print(key, self.data[key])
  # Setters

  def set_template(self, template):
    self.template = template

  def printl(self, line=""):
    if(self.verbose):
      print(self.st(), str(line))

  def log(self, line):
    #if(isinstance(line, (list,))):
    #  line = arr_to_line(line)
    if(not os.path.exists("log")):
      os.makedirs("log")
    fh = open("log/eoslog.txt", 'a')
    fh.write(line + "\n")
    fh.close()

  def make_dir(self, dir):
    if(not os.path.exists(dir)):
      os.makedirs(dir)
    return dir

  @staticmethod
  #def arr_to_line(str_in):
  #  if(isinstance(line, (list,))):

  @staticmethod
  def energies(file_list):
    e = []
    for file in file_list:
      pwo = pwscf_output(file)
      e.append(float(pwo.get_total_energy()))
    return e

  @staticmethod
  def volumes(file_list):
    v = []
    for file in file_list:
      pwo = pwscf_output(file)
      v.append(float(pwo.get_volume()))
    return v

  @staticmethod
  def energies(file_list):
    e = []
    for file in file_list:
      pwo = pwscf_output(file)
      e.append(float(pwo.get_total_energy()))
    return e

  @staticmethod
  def ve(file_list):
    v = []
    e = []
    for file in file_list:
      pwo = pwscf_output(file)
      v.append(float(pwo.get_volume()))
      e.append(float(pwo.get_total_energy()))
    return v, e

  @staticmethod
  def to_csv(filename, lists):
    lines = []
    for column in lists:
      for i in range(len(column)):
        if(i >= len(lines)):
          lines.append('')
        lines[i] += str(column[i]).strip() + ","
    fh = open(filename, "w")
    for i in range(len(lines)):
      fh.write(lines[i][0:-1] + "\n")
    fh.close()

  @staticmethod
  def make_points(x_in, c, type=None, points=101):
    x = []
    y = []
    if(type.lower() == "poly2" and len(c) == 3):
      for n in range(points):
        xi = x_in[0] + (x_in[-1] - x_in[0]) * (float(n) / float(points - 1))
        x.append(xi)
        y.append(c[2] + c[1] * float(xi) + c[0] * float(xi) **2)
    if(type.lower() == "eos"):
      for n in range(points):
        xi = x_in[0] + (x_in[-1] - x_in[0]) * (float(n) / float(points - 1))
        x.append(xi)
        y.append(pwscf_eos.bm_calc(xi, c))
    if(type.lower() == "isolated"):
      for n in range(points):
        xi = x_in[0] + (x_in[-1] - x_in[0]) * (float(n) / float(points - 1))
        x.append(xi)
        y.append(pwscf_eos.isolated_fit(c, xi))
    return x, y

  @staticmethod
  def cp_vol(cp):
    sa = cp[1][1] * cp[2][2] - cp[1][2] * cp[2][1]
    sb = cp[1][2] * cp[2][0] - cp[1][0] * cp[2][2]
    sc = cp[1][0] * cp[2][1] - cp[1][1] * cp[2][0]
    return (sa * cp[0][0] + sb * cp[0][1] + sc * cp[0][2])

  @staticmethod
  def bm_calc(V, eos):
    V0 = eos['V0']
    E0 = eos['E0']
    B0 = eos['B0']
    B0P = eos['B0P']
    eta = (V/V0)**(1/3.0)
    return E0 + (9/16.0) * (B0 * V0) * ((eta*eta - 1)*(eta*eta - 1)) * (6.0 + B0P * (eta * eta - 1) - 4 * eta * eta )

  @staticmethod
  def isolated_fit(p, x):
    return p[0] + p[1] * numpy.exp(p[2] * (x + p[3]))


# CLASS: PWSCF_INPUT
######################################

class pwscf_input:

  def __init__(self, file_name=None, file_dir=None):
    self.file_data = []
    self.file_name = None
    self.dir_name = None
    self.reset()
    self.defaults()
    if(file_name != None):
      self.load(file_name, file_dir)

  def reset(self):
    # Control
    self.control = {
      "calculation": None,
      "title": None,
      "verbosity": None,
      "restart_mode": None,
      "wf_collect": None,
      "nstep": None,
      "iprint": None,
      "tstress": None,
      "tprnfor": None,
      "dt": None,
      "outdir": None,
      "wfcdir": None,
      "prefix": None,
      "lkpoint_dir": None,
      "max_seconds": None,
      "etot_conv_thr": None,
      "forc_conv_thr": None,
      "disk_io": None,
      "pseudo_dir": None,
      "tefield": None,
      "dipfield": None,
      "lelfield": None,
      "nberrycyc": None,
      "lorbm": None,
      "lberry": None,
      "gdir": None,
      "nppstr": None,
      "lfcpopt": None,
      "gate": None
    }
    # SYSTEM
    self.system = {
      "ibrav": None,
      "celldm": None,
      "A": None,
      "B": None,
      "C": None,
      "cosAB": None,
      "cosAC": None,
      "cosBC": None,
      "nat": None,
      "ntyp": None,
      "nbnd": None,
      "tot_charge": None,
      "starting_charge": None,
      "tot_magnetization": None,
      "starting_magnetization": None,
      "ecutwfc": None,
      "ecutrho": None,
      "ecutfock": None,
      "nr1": None,
      "nr2": None,
      "nr3": None,
      "nr1s": None,
      "nr2s": None,
      "nr3s": None,
      "nosym": None,
      "nosym_evc": None,
      "noinv": None,
      "no_t_rev": None,
      "force_symmorphic": None,
      "use_all_frac": None,
      "occupations": None,
      "one_atom_occupations": None,
      "starting_spin_angle": None,
      "degauss": None,
      "smearing": None,
      "nspin": None,
      "noncolin": None,
      "ecfixed": None,
      "qcutz": None,
      "q2sigma": None,
      "input_dft": None,
      "exx_fraction": None,
      "screening_parameter": None,
      "exxdiv_treatment": None,
      "x_gamma_extrapolation": None,
      "ecutvcut": None,
      "nqx1": None,
      "nqx2": None,
      "nqx3": None,
      "lda_plus_u": None,
      "lda_plus_u_kind": None,
      "xdm": None,
      "xdm_a1": None,
      "xdm_a2": None,
      "space_group": None,
      "uniqueb": None,
      "origin_choice": None,
      "rhombohedral": None,
      "zgate": None,
      "relaxz": None,
      "block": None,
      "block_1": None,
      "block_2": None,
      "block_height": None
    }
    # ELECTRONS
    self.electrons = {
      "electron_maxstep": None,
      "scf_must_converge": None,
      "conv_thr": None,
      "adaptive_thr": None,
      "conv_thr_init": None,
      "conv_thr_multi": None,
      "mixing_mode": None,
      "mixing_beta": None,
      "mixing_ndim": None,
      "mixing_fixed_ns": None,
      "diagonalization": None,
      "ortho_para": None,
      "diago_thr_init": None,
      "diago_cg_maxiter": None,
      "diago_david_ndim": None,
      "diago_full_acc": None,
      "efield": None,
      "efield_cart": None,
      "efield_phase": None,
      "startingpot": None,
      "startingwfc": None,
      "tqr": None
    }
    # IONS
    self.ions = {
      "ion_dynamics": None,
      "ion_positions": None,
      "pot_extrapolation": None,
      "wfc_extrapolation": None,
      "remove_rigid_rot": None,
      "ion_temperature": None,
      "tempw": None,
      "tolp": None,
      "delta_t": None,
      "nraise": None,
      "refold_pos": None,
      "upscale": None,
      "bfgs_ndim": None,
      "trust_radius_max": None,
      "trust_radius_min": None,
      "trust_radius_ini": None,
      "w_1": None,
      "w_2": None
    }
    # CELL
    self.cell = {
      "cell_dynamics": None,
      "press": None,
      "wmass": None,
      "cell_factor": None,
      "press_conv_thr": None,
      "cell_dofree": None
    }
    # Other lists
    self.atomic_species = []
    self.atomic_positions = []
    self.k_points = []
    self.cell_parameters = []
    # File
    self.file = ""

  def defaults(self):
    try:
      self.scratch_dir = os.environ['PWSCF_SCRATCH']
    except:
      self.scratch_dir = '/opt/scratch'
    try:
      self.pp_dir = os.environ['PWSCF_PP']
    except:
      self.pp_dir = '/opt/pp'
  #  Load data from file

  def load(self, file_name, file_dir=None):
    self.file_name = file_name
    self.dir_name = file_dir
    if(file_dir != None):
      self.file_path = file_dir + "/" + file_name
    else:
      self.file_path = file_name
    data = self.load_from_file(self.file_path)
    self.load_data(data)
  # Load from a block of data (text, file etc)

  def load_data(self, data):
    # Store data into file_data list
    self.file_data.append(data)
    # Reset data store
    self.reset()
    # Clean
    data = pwscf_input.clean(data)
    # split
    data = data.split("\n")
    # Load keywords
    keywords = []
    # Load Keywords
    for line in data:
      line = line.strip()
      if(len(line)>0):
        # Remove trailing comma
        if(line[-1] == ","):
          line = line[0:-1]
        fields = line.split("=")
        if(len(fields) == 2):
          field_lc = fields[0].lower()
          keyword, id = pwscf_input.process_keyword(field_lc)
          pwscf_input.add_keyword(keywords, keyword, id, fields[1])
    for pair in keywords:
      if(pair[0] in self.control):
        self.control[pair[0]] = pair[1]
      elif(pair[0] in self.system):
        self.system[pair[0]] = pair[1]
      elif(pair[0] in self.electrons):
        self.electrons[pair[0]] = pair[1]
      elif(pair[0] in self.ions):
        self.ions[pair[0]] = pair[1]
      elif(pair[0] in self.cell):
        self.cell[pair[0]] = pair[1]
    # Load atomic species
    n_species = 0
    if(self.system['ntyp'] != None):
      try:
        n_species = int(self.system['ntyp'])
      except:
        n_species = 0
    if(n_species > 0):
      counter = 0
      for line in data:
        line = line.strip()
        if(line.upper()[0:14] == "ATOMIC_SPECIES"):
          counter = counter + 1
        elif(counter > 0 and counter <= n_species and line != ""):
          counter = counter + 1
          self.atomic_species.append(pwscf_input.fields(line))
    # Load atomic positions
    n_atoms = 0
    if(self.system['nat'] != None):
      try:
        n_atoms = int(self.system['nat'])
      except:
        n_atoms = 0
    if(n_atoms > 0):
      counter = 0
      for line in data:
        line = line.strip()
        if(line.upper()[0:16] == "ATOMIC_POSITIONS"):
          fields = pwscf_input.fields(line)
          if(len(fields) == 2):
            self.atomic_positions.append(fields[1])
          counter = counter + 1
        elif(counter > 0 and counter <= n_atoms and line != ""):
          counter = counter + 1
          self.atomic_positions.append(pwscf_input.fields(line))
    # k_points
    flag = 0
    for line in data:
      line = line.strip()
      if(line.upper()[0:8] == "K_POINTS"):
        fields = pwscf_input.fields(line)
        k_points_type = fields[1]
        self.k_points.append(k_points_type)
        if(k_points_type.upper() == "AUTOMATIC"):
          flag = 1
      elif(flag > 0):
        flag = flag - 1
        fields = pwscf_input.fields(line)
        self.k_points.append(fields)
    # cell parameters
    flag = 0
    for line in data:
      line = line.strip()
      if(line.upper()[0:15] == "CELL_PARAMETERS"):
        fields = pwscf_input.fields(line)
        self.cell_parameters.append(fields[1])
        flag = 3
      elif(flag>0):
        fields = pwscf_input.fields(line)
        self.cell_parameters.append(fields)
    self.make()
  #  Run as it's own program

  def run(self):
    self.reset()
    option = ""
    file_name = ""
    if(len(sys.argv) > 1 and sys.argv[1] is not None):
      option = sys.argv[1]
    if(len(sys.argv) > 2 and sys.argv[2] is not None):
      file_name = sys.argv[2]
    if(option.lower().strip() == "" or option.lower().strip() == "interactive"):
      self.menu()
      exit()
    elif(option.lower().strip() == "quiet"):
      print("Quiet")
    else:
      return 0
# READ/LOAD input file

  def load_from_file(self, file_name):
    # Init variable
    file_data = ""
    # Read it in line by line
    fh = open(file_name, "r")
    for file_row in fh:
      file_data = file_data + file_row.strip() + '\n'
    return file_data
# MAKE input file

  def make(self):
    now = datetime.datetime.now()
    time_now = str(now.hour) + ":" + str(now.minute) + "   " + str(now.day) + "/" + str(now.month) + "/" + str(now.year)
    file = "! Edited " + time_now + "\n"
    # CONTROL
    file += "&CONTROL \n"
    for key in sorted(self.control.keys()):
      file += pwscf_input.make_line(key, self.control[key])
    file += "/ \n"
    # SYSTEM
    file += "&SYSTEM \n"
    for key in sorted(self.system.keys()):
      file += pwscf_input.make_line(key, self.system[key])
    file += "/ \n"
    # ELECTRONS
    file += "&ELECTRONS \n"
    for key in sorted(self.electrons.keys()):
      value = self.electrons[key]
      if(value != None):
        file += key + " = " + value + ", \n"
    file += "/ \n"
    # IONS
    file += "&IONS \n"
    for key in sorted(self.ions.keys()):
      value = self.ions[key]
      if(value != None):
        file += key + " = " + value + ", \n"
    file += "/ \n"
    # CELL
    file += "&CELL \n"
    for key in sorted(self.cell.keys()):
      value = self.cell[key]
      if(value != None):
        file += key + " = " + value + ", \n"
    file += "/ \n"
    # ATOMIC_SPECIES
    file += "ATOMIC_SPECIES \n"
    for species in self.atomic_species:
      for field in species:
        file += field + " "
      file += "\n"
    # ATOMIC_POSITIONS
    header = 0
    for position in self.atomic_positions:
      if(header == 0):
        file += "ATOMIC_POSITIONS "
        file += position + "\n"
        header = 1
      #elif(header == 1):
      #  file += position[1] + "\n"
      #  header = 2
      elif(header == 1):
        for field in position:
          file += field + "   "
        file += "\n"
    # K_POINTS
    file += "K_POINTS " + self.k_points[0]
    file += "\n"
    for i in range(1,len(self.k_points)):
      for point in self.k_points[i]:
        file += point + " "
      file += "\n"
    # K_POINTS
    file += "CELL_PARAMETERS " + self.cell_parameters[0]
    file += "\n"
    for i in range(1,len(self.cell_parameters)):
      for point in self.cell_parameters[i]:
        file += point + " "
      file += "\n"
    # Process
    file = file.strip()
    # Store data into file_data list
    self.file_data.append(file)

  def print_out(self):
    self.make()
    print(self.file_data[-1])

  def print_history(self):
    for file in self.file_data:
      print(file)
      print()
#  Save, Save/Load Original

  def save(self, file=None, dir=None):
    # Build latest version of file
    self.make()
    if(file == None):
      file = self.file_name
    if(dir == None):
      dir = self.dir_name
    self.file_name = file
    self.dir_name = dir
    if(dir == None):
      path = file
    else:
      if (not os.path.exists(dir)):
        os.makedirs(dir)
      path = dir + "/" + file
    # Write latest data
    fh = open(path, "w")
    fh.write(self.file_data[-1])
    fh.close()

  def save_original(self, file=None, dir=None):
    if(file == None):
      file = self.file_name
    if(dir == None):
      dir = self.dir_name
    self.file_name = file
    self.dir_name = dir
    if(dir == None):
      path = file
    else:
      if (not os.path.exists(dir)):
        os.makedirs(dir)
      path = dir + "/" + file
    # Write latest data
    fh = open(path, "w")
    fh.write(self.file_data[0])
    fh.close()

  def load_original(self, file=None, dir=None):
    self.load_data(self.file_data[0])
#  Set
  # Control

  def set_calculation(self, type=None):
    if(type == None):
      type = "SCF"
    self.control['calculation'] = '"' + type.lower() + '"'

  def set_var(self, var, value):
    for key in sorted(self.control.keys()):
      if(key.upper().strip() == var.upper().strip()):
        self.control['key'] = value
        return True
    return False

  def set_dirs(self):
    self.control['outdir'] = '"' + self.scratch_dir + '"'
    self.control['pseudo_dir'] = '"' + self.pp_dir + '"'

  def set_prefix(self, pin = None):
    if(pin == None):
      pin = pwscf_input.rand_string()
    self.control['prefix'] = '"' + pin + '"'
  # System

  def set_alat(self, alat):
    self.system['celldm'][0] = str(alat)

  def set_ecutrho(self, ecutrho):
    self.system['ecutrho'] = str(ecutrho)

  def set_ecutwfc(self, ecutwfc):
    self.system['ecutwfc'] = str(ecutwfc)

  def set_nosym(self, nosym=False):
    if(nosym == True or nosym.lower() == ".true."):
      self.system['nosym'] = ".TRUE."
    else:
      self.system['nosym'] = ".FALSE."

  def set_nspin(self, value):
    if(value == 2 or value == "2"):
      self.system['nspin'] = "2"
    elif(value == 4 or value == "4"):
      self.system['nspin'] = "4"
    else:
      self.system['nspin'] = "1"

  def set_tot_magnetization(self, tot_magnetization):
    self.system['tot_magnetization'] = str(tot_magnetization)

  def set_as_isolated(self, alat=10.0):
    self.set_alat(alat)
    self.set_nosym(True)
    self.set_nspin(2)
    self.set_tot_magnetization(0)
    self.set_cp_identity()
    self.load_config("ISOLATED")
  # Cell Parameters
  # Just the 3x3 array

  def set_cell_parameters(self, cell_in):
    type = self.cell_parameters[0]
    self.cell_parameters = [type]
    for row in cell_in:
      new_row = []
      for cell in row:
        new_row.append(str(cell))
      self.cell_parameters.append(new_row)
  # Just the 3x3 array

  def set_cp_arr(self, cp):
    self.set_cell_parameters(cp)
  # Copy entire list [,[,,],[,,],[,,]]

  def set_cp(self, cp):
    self.cell_parameters = cp

  def set_cp_identity(self):
    self.set_cell_parameters([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

  def nomalise_cell_parameters(self):
    self.system['celldm'][0] = str(float(self.system['celldm'][0]) * float(self.cell_parameters[1][0]))
    d = float(self.cell_parameters[1][0])
    for i in range(1,4):
      for j in range(0,3):
        self.cell_parameters[i][j] = str(float(self.cell_parameters[i][j]) / d)
#  Load Atom Config

  def load_config(self, type="FCC", size=1, alat=None, cp=None):
    type = type.upper()
    # FCC
    if(type == "FCC"):
      labels = []
      for row in self.atomic_species:
        labels.append(row[0])
      atoms = pwscf_standard.fcc(labels,size)
      self.atomic_positions = atoms
      self.system['nat'] = str(len(self.atomic_positions) - 1)
    # ISOLATED
    if(type == "ISOLATED"):
      labels = []
      for row in self.atomic_species:
        labels.append(row[0])
      atoms = pwscf_standard.isolated(labels)
      self.atomic_positions = atoms
      self.system['nat'] = str(len(self.atomic_positions) - 1)
    # ALAT
    if(alat != None):
      self.set_alat(alat)
    # CP
    if(cp != None):
      self.set_cp_arr(cp)
    self.make()
#  Get

  def get_path(self):
    if(self.file_name == None):
      file = "pwscf.in"
    else:
      file = self.file_name
    if(self.dir_name == None):
      path = file
    else:
      path = self.dir_name + "/" + file
    return path

  def get_file_name(self):
    return self.file_name

  def get_cp_array(self):
    cp = numpy.zeros((3,3))
    for i in range(3):
      for j in range(3):
        cp[i,j] = float(self.cell_parameters[i+1][j])
    return cp

  def get_data(self, make=False):
    if(make):
      self.make()
    return self.file_data[-1]

  def get_nat(self):
    return int(self.system['nat'])
# Signature

  def signature(self):
    # CONTROL
    file = "&CONTROL \n"
    for key in sorted(self.control.keys()):
      read = True
      if(key == "outdir"):
        read = False
      if(key == "prefix"):
        read = False
      if(read):
        file += pwscf_input.make_line(key, self.control[key])
    file += "/ \n"
    # SYSTEM
    file += "&SYSTEM \n"
    for key in sorted(self.system.keys()):
      file += pwscf_input.make_line(key, self.system[key])
    file += "/ \n"
    # ELECTRONS
    file += "&ELECTRONS \n"
    for key in sorted(self.electrons.keys()):
      value = self.electrons[key]
      if(value != None):
        file += key + " = " + value + ", \n"
    file += "/ \n"
    # IONS
    file += "&IONS \n"
    for key in sorted(self.ions.keys()):
      value = self.ions[key]
      if(value != None):
        file += key + " = " + value + ", \n"
    file += "/ \n"
    # CELL
    file += "&CELL \n"
    for key in sorted(self.cell.keys()):
      value = self.cell[key]
      if(value != None):
        file += key + " = " + value + ", \n"
    file += "/ \n"
    # ATOMIC_SPECIES
    file += "ATOMIC_SPECIES \n"
    for species in self.atomic_species:
      for field in species:
        file += str(field) + " "
      file += "\n"
    # ATOMIC_POSITIONS
    header = 0
    for position in self.atomic_positions:
      if(header == 0):
        file += "ATOMIC_POSITIONS "
        file += position + "\n"
        header = 1
      elif(header == 1):
        for field in position:
          file += str(field) + "   "
        file += "\n"
    # K_POINTS
    file += "K_POINTS " + self.k_points[0]
    file += "\n"
    for i in range(1,len(self.k_points)):
      for point in self.k_points[i]:
        file += point + " "
      file += "\n"
    # K_POINTS
    file += "CELL_PARAMETERS " + self.cell_parameters[0]
    file += "\n"
    for i in range(1,len(self.cell_parameters)):
      for point in self.cell_parameters[i]:
        file += point + " "
      file += "\n"
    # String being hashed must be converted to utf-8 encoding
    input_string = file.encode('utf-8')
    # Make hash object
    my_hash = hashlib.sha512()
    # Update
    my_hash.update(input_string)
    # Return hash
    return my_hash.hexdigest()
# Interactive

  def menu(self):
    while(True):
      choice = self.print_menu().upper()
      print(choice)
      if(choice == "X"):
        exit()
      elif(choice == "1"):
        self.i_load()
      elif(choice == "2"):
        self.i_display()

  def print_menu(self):
    pwscf_input.header("Menu")
    print("1. Load File")
    print("2. Display File")
    print("X. Exit")
    return input("Choice: ")

  def i_load(self):
    pwscf_input.header("Load Input File")
    file_name = input("Enter file name: ")
    self.load(file_name)
    input()

  def i_display(self):
    self.make()
    pwscf_input.header("Display File")
    print(self.file)
    input()
# Help

  def help(self):
    print("HELP")
# Static Methods

  @staticmethod
  def remove_spaces(input_string):
    return input_string.replace(" ", "")

  @staticmethod
  def fields(input_string):
    input_string = input_string.strip()
    output_string = ""
    last = None
    for character in input_string:
      if(character != " " or (character == " " and last != " ")):
        output_string += character
    return output_string.split(" ")

  @staticmethod
  def check_keyword(line, keyword):
    if(line.upper()[0:len(keyword)] == keyword.upper()):
      return True
    return False

  @staticmethod
  def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

  @staticmethod
  def header(sub_title=""):
    pwscf_input.clear_screen()
    print("==========================================================")
    print("                    PWscf Input Editor                    ")
    print("==========================================================")
    print()
    print(sub_title)
    print()
    print()
    print()

  @staticmethod
  def process_keyword(str_in):
    str_in = str_in.lower().strip()
    str_in = pwscf_input.remove_spaces(str_in)
    id = None
    keyword = ""
    flag = 0
    for character in str_in:
      if(character == "("):
        id = ""
        flag = 1
      elif(character == ")"):
        flag = 2
      elif(flag == 0):
        keyword += character
      elif(flag == 1):
        id = id + character
    if(id != None):
      try:
        id = int(id)
      except:
        id = None
    return keyword, id

  @staticmethod
  def add_keyword(keywords, keyword, id, value):
    if(id == None):
      added = False
      for i in range(len(keywords)):
        if(keywords[i][0] == keyword):
          added = True
          keywords[i][1] = keyword
      if(added == False):
        keywords.append([keyword, value])
    else:
      n = None
      for i in range(len(keywords)):
        if(keywords[i][0] == keyword):
          n = i
          break
      if(n == None):
        keywords.append([keyword,[None]])
        n = len(keywords) - 1
      while(len(keywords[n][1]) < id):
        keywords[n][1].append(None)
      keywords[n][1][id-1] = value

  @staticmethod
  def make_line(key, value):
    output = ""
    if(value != None):
       if(isinstance(value, (list,))):
         for i in range(len(value)):
           if(value[i] != None):
             output += key + "(" + str(i+1) + ") = " + value[i] + ", \n"
       else:
         output += key + " = " + value + ", \n"
    return output

  @staticmethod
  def coord_format(float_in):
    pad = "              "
    value = str(round(float_in, 6)).strip()
    return value

  @staticmethod
  def label_format(label):
    pad = "              "
    label = label.strip()
    return label

  @staticmethod
  def clean(str_in):
    str_out = ""
    l = len(str_in)
    for i in range(l):
      # Last, Next, This
      if(i == 0):
        last = None
      else:
        last = str_in[i-1]
      if(i < (l-1)):
        next = str_in[i+1]
      else:
        next = None
      char = str_in[i]
      # Check
      ok = True
      if(last == " " and char == " "):
        ok = False
      elif(last == "\n" and char == "\n"):
        ok = False
      elif(last == "\n" and char == " "):
        ok = False
      elif(char == " " and next == "\n"):
        ok = False
      elif(last == "=" and char == " "):
        ok = False
      elif(char == " " and next == "="):
        ok = False
      # Add to string
      if(ok):
        str_out += char
    return str_out

  @staticmethod
  def rand_string(len_in=16):
    output = ""
    char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOQRSTUVWXYZ"
    for i in range(len_in):
      r = random.randint(0,len(char_set)-1)
      output += char_set[r]
    return output
        #data = re.sub(r'\s\s+', ' ', data)
    #data = re.sub(r'\s=\s', '=', data)


# CLASS: PWSCF_STANDARD
######################################

class pwscf_standard:

  @staticmethod
  def fcc(label, size=1):
    if(not isinstance(label, (list,))):
      label = [label]
    atoms = ['crystal']
    for x in range(size):
      for y in range(size):
        for z in range(size):
          coords = [[str((x+0.0)/size), str((y+0.0)/size), str((z+0.0)/size)],
                    [str((x+0.5)/size), str((y+0.5)/size), str((z+0.0)/size)],
                    [str((x+0.5)/size), str((y+0.0)/size), str((z+0.5)/size)],
                    [str((x+0.0)/size), str((y+0.5)/size), str((z+0.5)/size)]]
          for i in range(len(coords)):
            atoms.append([label[i % len(label)],coords[i][0],coords[i][1],coords[i][2]])
    return atoms

  @staticmethod
  def isolated(label, size=1):
    if(not isinstance(label, (list,))):
      label = [label]
    atoms = ['crystal']
    atoms.append([label[0], "0.5", "0.5", "0.5"])
    return atoms


# CLASS: PWSCF_OUTPUT
######################################

class pwscf_output:

  def __init__(self, file_in=None):
    self.reset()
    if(file_in != None):
      self.load(file_in)

  def reset(self):
    self.z = numpy.zeros((3,3))
    # Control
    self.data = {
      "ok": False,
      "job_done": False,
      "error": False,
      "type": None,
      "summary": None,
      "mpi_processes": None,
      "bravais_lattice_index": None,
      "alat": None,
      "volume": None,
      "electrons": None,
      "electrons_up": None,
      "electrons_down": None,
      "crystal_in": numpy.zeros((3,3)),
      "crystal_calc": numpy.zeros((3,3)),
      "total_energy": None,
      "density_full": None,
      "density": None,
      "cpu_time": None,
      "wall_time": None
    }
  #  Load, and use in another program

  def load(self, file_name):
    # Load data from file
    data = self.load_from_file(file_name)
    # Read through data
    self.load_data(data)
  # Load from a block of data (text, file etc)

  def load_data(self, data):
    # Reset data store
    self.reset()
    # split
    data = data.split("\n")
    # OK
    self.data['ok'] = False
    counter = 0
    for line in data:
      line = line.strip()
      if(pwscf_output.compare(line, "JOB DONE.")):
        counter = counter + 1
      if(pwscf_output.compare(line, "Exit code:")):
        counter = counter - 1
      if(pwscf_output.compare(line, "convergence NOT achieved")):
        counter = counter - 1
    if(counter == 1):
      self.data['ok'] = True
    # Calc Type
    self.data['type'] = "SCF"
    for line in data:
      line = line.strip()
      if(line[0:23] == "A final scf calculation"):
        self.data['type'] = "VC-RELAX"
    # Load
    n = 0
    counter = 0
    while(n < len(data)):
      n, line, line_uc = self.next_line(n, data)
      if(line != ""):
        counter += 1
        if(counter == 1):
          self.data['summary'] = line
        else:
          if(pwscf_output.compare(line, "Number of MPI processes:")):
            self.data['mpi_processes'] = pwscf_output.extract(line, ":", "", "i")
          if(pwscf_output.compare(line, "bravais-lattice index     =")):
            self.data['bravais_lattice_index'] = pwscf_output.extract(line, "=", "", "i")
          if(pwscf_output.compare(line, "lattice parameter (alat)  =")):
            self.data['alat'] = pwscf_output.extract(line, "=", "a.u.", "f")
          if(pwscf_output.compare(line, "unit-cell volume          =")):
            self.data['volume'] = pwscf_output.extract(line, "=", "(a.u.)^3", "f")
          if(pwscf_output.compare(line, "number of atoms/cell      =")):
            self.data['nat'] = pwscf_output.extract(line, "=", "", "i")
          if(pwscf_output.compare(line, "number of atomic types    =")):
            self.data['types'] = pwscf_output.extract(line, "=", "", "i")
          if(pwscf_output.compare(line, "number of electrons       =")):
            str_e = pwscf_output.extract(line, "=", "", "s")
            e, eu, ed = pwscf_output.electron_string(str_e)
            self.data['electrons'] = e
            self.data['electrons_up'] = eu
            self.data['electrons_down'] = ed
          if(pwscf_output.compare(line, "number of Kohn-Sham states=")):
            self.data['ks_states'] = pwscf_output.extract(line, "=", "", "i")
          if(pwscf_output.compare(line, "kinetic-energy cutoff     =")):
            self.data['ecutwfc'] = pwscf_output.extract(line, "=", "Ry", "f")
          if(pwscf_output.compare(line, "charge density cutoff     =")):
            self.data['ecutrho'] = pwscf_output.extract(line, "=", "Ry", "f")
          if(pwscf_output.compare(line.strip(), "crystal axes:") and pwscf_output.is_zero(self.data['crystal_in'])):
            for j in range(3):
              n, line, line_uc = self.next_line(n, data)
              fields = pwscf_output.extract(line, "= (", ")", "s", " ")
              self.data['crystal_in'][j,:] = fields
              self.data['crystal_calc'][j,:] = fields
          if(pwscf_output.compare(line.strip(), "crystal axes:")):
            for j in range(3):
              n, line, line_uc = self.next_line(n, data)
              fields = pwscf_output.extract(line, "= (", ")", "s", " ")
              self.data['crystal_calc'][j,:] = fields
          if(pwscf_output.compare(line, "!    total energy")):
            self.data['total_energy'] = pwscf_output.extract(line, "=", "Ry", "f")
          if(pwscf_output.compare(line, "Total force =")):
            self.data['total_force'] = pwscf_output.extract(line, "=", "T", "f")
          if(pwscf_output.compare(line, "density = ")):
            self.data['density_full'] = pwscf_output.extract(line, "=", "", "s")
            self.data['density'] = pwscf_output.extract(line, "=", "g/cm^3", "f")
          if(pwscf_output.compare(line, "PWSCF        :")):
            self.data['cpu_time'] = pwscf_output.extract(line, ":", "CPU", "s")
          if(pwscf_output.compare(line, "PWSCF        :")):
            self.data['wall_time'] = pwscf_output.extract(line, "CPU", "WALL", "s")
          if(pwscf_output.compare(line, "JOB DONE.")):
            self.data['job_done'] = True
          if(pwscf_output.compare(line, "Exit code:")):
            self.data['error'] = True

  def next_line(self, n, data):
    if(n < len(data)):
      line = data[n].strip()
      line_uc = line.upper()
      n = n + 1
      return n, line, line_uc
    else:
      n = n + 1
      return n, None, None

  def store(self, store, line, field, n=0):
    l, f = pwscf_output.read_line(line, field)
    if(l != False):
      self.data[store] = f[n]
  #  Run as it's own program

  def run(self):
    self.reset()
    option = ""
    file_name = ""
    if(len(sys.argv) > 1 and sys.argv[1] is not None):
      option = sys.argv[1]
    if(len(sys.argv) > 2 and sys.argv[2] is not None):
      file_name = sys.argv[2]
    if(option.lower().strip() == "" or option.lower().strip() == "interactive"):
      self.menu()
      exit()
    elif(option.lower().strip() == "quiet"):
      print("Quiet")
    else:
      return 0
# READ/LOAD input file

  def load_from_file(self, file_name):
    # Init variable
    file_data = ""
    # Read it in line by line
    fh = open(file_name, "r")
    for file_row in fh:
      file_data = file_data + file_row.strip() + '\n'
    return file_data
# Get

  def get_alat(self):
    return self.data['alat']

  def get_volume(self):
    return self.data['volume']

  def get_total_energy(self):
    return self.data['total_energy']

  def get_density(self):
    return self.data['density']

  def get_cell_parameters(self):
    cp = ['alat',
          [str(self.data['crystal_calc'][0,0]), str(self.data['crystal_calc'][0,1]), str(self.data['crystal_calc'][0,2])],
          [str(self.data['crystal_calc'][1,0]), str(self.data['crystal_calc'][1,1]), str(self.data['crystal_calc'][1,2])],
          [str(self.data['crystal_calc'][2,0]), str(self.data['crystal_calc'][2,1]), str(self.data['crystal_calc'][2,2])]]
    return cp

  def get_job_done(self):
    return self.data['job_done']

  def get_ok(self):
    return self.data['ok']
# Interactive

  def menu(self):
    while(True):
      choice = self.print_menu().upper()
      print(choice)
      if(choice == "X"):
        exit()
      elif(choice == "1"):
        self.i_load()
      elif(choice == "2"):
        self.i_display()

  def print_menu(self):
    pwscf_output.header("Menu")
    print("1. Load File")
    print("2. Display File")
    print("X. Exit")
    return input("Choice: ")

  def i_load(self):
    pwscf_output.header("Load Output File")
    file_name = input("Enter file name: ")
    self.load(file_name)
    print("File loaded.")
    input()

  def i_display(self):
    pwscf_output.header("Display File")
    self.output_details()
    input()

  def output_details(self):
    print("Output")
    for key in sorted(self.data.keys()):
      value = self.data[key]
      print(key, ":  ", value)
    #for key, value in self.data.items():
    #  print(key, ":  ", value)
# Static Methods

  @staticmethod
  def remove_spaces(input_string):
    return input_string.replace(" ", "")

  @staticmethod
  def extract(input_string, start=None, end=None, type=None, split=None):
    if(start == ""):
      start = None
    if(end == ""):
      end = None
    # Start/End
    start_n = None
    end_n = None
    if(start == None and end == None):
      start_n = 0
      end_n = len(input_string)
    elif(start == None and end != None):
      end_l = len(end)
      start_n = 0
      for n in range(len(input_string)):
        if(input_string[n:n+end_l] == end[0:end_l]):
          end_n = n
          break
    elif(start != None and end == None):
      start_l = len(start)
      end_n = len(input_string)
      for n in range(len(input_string)):
        if(input_string[n:n+start_l] == start[0:start_l]):
          start_n = n + start_l
    else:
      start_l = len(start)
      end_l = len(end)
      for n in range(len(input_string)):
        if(input_string[n:n+start_l] == start[0:start_l]):
          start_n = n + start_l
        if(start_n != None and input_string[n:n+end_l] == end[0:end_l]):
          end_n = n
          break
    # Read
    result = input_string[start_n:end_n].strip()
    if(type.lower() == "f"):
      result = float(result)
    elif(type.lower() == "i"):
      result = int(result)
    # Split
    if(split != None):
      if(split == " "):
        result = re.sub(r'\s\s+', ' ', result)
      result = result.split(split)
    # Return
    return result

  @staticmethod
  def compare(line, field):
    line = line.strip()
    line = line.upper()
    field = field.strip()
    field = field.upper()
    f_len = len(field)
    if(len(line) >= f_len and line[0:f_len] == field[0:f_len]):
      return True
    return False

  @staticmethod
  def read_line(line, field):
    line = line.strip()
    #line = re.sub(r'\s\s+', ' ', line)
    #line = re.sub(r'\s=\s', '=', line)
    line = pwscf_output.clean(line)
    line_uc = line.upper()
    field = field.strip()
    #field = re.sub(r'\s\s+', ' ', field)
    #field = re.sub(r'\s=\s', '=', field)
    field = pwscf_output.clean(field)
    field = field.upper()
    f_len = len(field)
    if(len(line_uc) >= f_len and line_uc[0:f_len] == field[0:f_len]):
      output = line[f_len:].strip()
      fields = output.split(" ")
      return output, fields
    return False, False

  @staticmethod
  def fields(input_string):
    input_string = input_string.strip()
    output_string = ""
    last = None
    for character in input_string:
      if(character != " " or (character == " " and last != " ")):
        output_string += character
    return output_string.split(" ")

  @staticmethod
  def check_keyword(line, keyword):
    if(line.upper()[0:len(keyword)] == keyword.upper()):
      return True
    return False

  @staticmethod
  def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

  @staticmethod
  def header(sub_title=""):
    pwscf_output.clear_screen()
    print("==========================================================")
    print("                    PWscf Input Editor                    ")
    print("==========================================================")
    print()
    print(sub_title)
    print()
    print()
    print()

  @staticmethod
  def process_keyword(str_in):
    str_in = str_in.lower().strip()
    str_in = pwscf_output.remove_spaces(str_in)
    id = None
    keyword = ""
    flag = 0
    for character in str_in:
      if(character == "("):
        id = ""
        flag = 1
      elif(character == ")"):
        flag = 2
      elif(flag == 0):
        keyword += character
      elif(flag == 1):
        id = id + character
    if(id != None):
      try:
        id = int(id)
      except:
        id = None
    return keyword, id

  @staticmethod
  def add_keyword(keywords, keyword, id, value):
    if(id == None):
      added = False
      for i in range(len(keywords)):
        if(keywords[i][0] == keyword):
          added = True
          keywords[i][1] = keyword
      if(added == False):
        keywords.append([keyword, value])
    else:
      n = None
      for i in range(len(keywords)):
        if(keywords[i][0] == keyword):
          n = i
          break
      if(n == None):
        keywords.append([keyword,[None]])
        n = len(keywords) - 1
      while(len(keywords[n][1]) < id):
        keywords[n][1].append(None)
      keywords[n][1][id-1] = value

  @staticmethod
  def make_line(key, value):
    output = ""
    if(value != None):
       if(isinstance(value, (list,))):
         for i in range(len(value)):
           if(value[i] != None):
             output += key + "(" + str(i+1) + ") = " + value[i] + ", \n"
       else:
         output += key + " = " + value + ", \n"
    return output

  @staticmethod
  def coord_format(float_in):
    pad = "              "
    value = str(round(float_in, 6)).strip()
    return value

  @staticmethod
  def label_format(label):
    pad = "              "
    label = label.strip()
    return label

  @staticmethod
  def is_zero(arr):
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        if(arr[i, j] != 0.0):
          return False
    return True

  @staticmethod
  def clean(str_in):
    str_out = ""
    l = len(str_in)
    for i in range(l):
      # Last, Next, This
      if(i == 0):
        last = None
      else:
        last = str_in[i-1]
      if(i < (l-1)):
        next = str_in[i+1]
      else:
        next = None
      char = str_in[i]
      # Check
      ok = True
      if(last == " " and char == " "):
        ok = False
      elif(last == "\n" and char == "\n"):
        ok = False
      elif(last == "\n" and char == " "):
        ok = False
      elif(char == " " and next == "\n"):
        ok = False
      elif(last == "=" and char == " "):
        ok = False
      elif(char == " " and next == "="):
        ok = False
      # Add to string
      if(ok):
        str_out += char
    return str_out

  @staticmethod
  def electron_string(str_in):
    arr = str_in.split("(up:")
    e = arr[0]
    if(len(arr) == 1):
      return e.strip(), None, None
    if(len(arr)==2):
      arr_b = arr[1].split(", down:")
      eu = arr_b[0]
      arr_c = arr_b[1].split(")")
      ed = arr_c[0]
      return e.strip(), eu.strip(), ed.strip()
    print("TEST")
    return "","",""


# CLASS: PWSCF_EXEC
######################################

class pwscf_exec:

  def __init__(self, file=None, dir=None, cache=True, random_prefix=True):
    self.cache = cache
    self.random_prefix = random_prefix
    self.reset()
    self.add_file(file, dir)

  def reset(self):
    self.log_set = False
    self.tae = False
    self.set_procs()
    self.set_bin()
    self.set_cache()
    self.files = []
    self.exec_dir = None

  def set_terminate_at_error(self, value = True):
    self.tae = value

  def set_exec_dir(self, dir=None):
    if(dir != None):
      if (not os.path.exists(dir)):
        os.makedirs(dir)
      self.exec_dir = dir

  def set_procs(self):
    try:
      self.procs = os.environ['PROC_COUNT']
    except:
      self.procs = 40
    self.log("Procs: " + str(self.procs))

  def set_bin(self):
    try:
      self.pw = os.environ['PWSCF_BIN']
    except:
      self.pw = "/rds/homes/b/bxp912/apps/qe-6.3/bin/pw.x"
    self.log("Bin: " + str(self.pw))

  def set_cache(self):
    try:
      self.cache = os.environ['PWSCF_CACHE']
    except:
      self.cache = ".cache"
    self.log("Cache: " + str(self.cache))

  def add_file(self, file_in, dir = None):
    output = []
    file_path = None
    if(file_in != None):
      if(isinstance(file_in, (list,))):
        for file in file_in:
          if(dir == None):
            file_path = file
          else:
            file_path = dir + "/" + file
          output.append(self.add_each_file(file_path))
      else:
        if(dir == None):
          file_path = file_in
        else:
          file_path = dir + "/" + file_in
        output.append(self.add_each_file(file_path))
    # Return input and output file names
    return output

  def add_each_file(self, file_path):
    file_path = file_path.strip()
    if(os.path.isfile(file_path)):
      file_in = pwscf_input(file_path)
      file_in.set_dirs()
      if(self.random_prefix):
        file_in.set_prefix()
      file_in.save(None, self.exec_dir)
      details = {
                 "in": file_in.get_path(),
                 "out": pwscf_exec.fno(file_in.get_path()),
                 "sig": file_in.signature()
      }
      self.files.append(details)
      return details

  def read_arg(self):
    if(len(sys.argv) >= 2 and sys.argv[1] != None):
      list_file = sys.argv[1]
      if(os.path.isfile(list_file)):
        fh = open(list_file, 'r')
        for line in fh:
          in_file = line.strip()
          if(os.path.isfile(in_file)):
            self.add_file(in_file)
    if(len(sys.argv) >= 3 and sys.argv[2] != None):
      self.set_exec_dir(sys.argv[2])

  def run(self):
    self.pw_in = []
    self.pw_out = []
    for file in self.files:
      #file_in = pwscf_input(file.strip())
      #file_in.set_dirs()
      #if(self.random_prefix):
      #  file_in.set_prefix()
      #file_in.save(None, self.exec_dir)
      #sig = file_in.signature()
      self.pw_in.append(file['in'])
      cmd = "mpirun -n " + str(self.procs) + " " + self.pw + " -i " + self.pw_in[-1] + " > " + file['out']
      self.log("Cmd: " + cmd)
      # Make cache dir
      if (not os.path.exists(self.cache)):
        os.makedirs(self.cache)
      # Always cache input file
      copyfile(self.pw_in[-1], self.cache + "/" + file['sig'] + ".in")
      #print(self.pw_in[-1])
      #print(sig)
      # Used cache output
      run = True
      if(self.cache):
        run = False
        cache_out = self.cache + "/" + file['sig'] + ".out"
        if(os.path.isfile(cache_out)):
          pwo = pwscf_output(cache_out)
          if(pwo.get_job_done()):
            copyfile(self.cache + "/" + file['sig'] + ".out", file['out'])
            print("Used cache")
          else:
            run = True
        else:
          run = True
      else:
        run = True
      if(run):
        # If no cache, just run
        os.system(cmd)
        pwo = pwscf_output(file['out'])
        if(pwo.get_job_done()):
          copyfile(file['out'], self.cache + "/" + file['sig'] + ".out")
      # Check output
      if(pwo.get_ok()):
        self.pw_out.append(file['out'])
      else:
        print("failed: ",file['out'])
        if(self.tae):
          return self.pw_out
    return self.pw_out

  def log(self, line):
    if (not os.path.exists("log")):
      os.makedirs("log")
    if(self.log_set == False):
      self.log_set = True
      fh = open("log/execlog.txt", 'w')
      fh.write('')
      fh.close()
    if(not os.path.exists("log")):
      os.makedirs("log")
    fh = open("log/execlog.txt", 'a')
    fh.write(line + "\n")
    fh.close()
  # Get

  def get_output_files(self):
    return self.pw_out

  @staticmethod
  def fno(fni):
    return fni.replace(".in", ".out")

  @staticmethod
  def hashname(file_path):
    data = ""
    fh = open(file_path, "r")
    for line in fh:
      data += line
    fh.close()
    return pwscf_exec.hash(data)

  @staticmethod
  def hash(input_string):
    # String being hashed must be converted to utf-8 encoding
    input_string = input_string.encode('utf-8')
    # Make hash object
    my_hash = hashlib.sha512()
    # Update
    my_hash.update(input_string)
    # Return hash
    return my_hash.hexdigest()


# CLASS: LMA
######################################

class lma:

  def __init__(self, file_input = None, verbose = False):
    self.reset()
    self.verbose = verbose
    if(file_input != None):
      if(isinstance(file_input, (list,))):
        if(len(file_input) > 2 or (len(file_input) == 2 and len(file_input[0]) == 2)):
          self.set_data_a(file_input)
        elif(len(file_input) == 2 and len(file_input[0]) > 2):
          self.set_data_b(file_input)
      elif(isinstance(file_input,numpy.ndarray)):
        self.set_data_c(file_input)
      elif(os.path.isfile(file_input)):
        self.load_file(file_input)
  #  Defaults

  def reset(self):
    self.verbose = False
    self.conv_thr = 1.0E-9
    self.max_cycles = 10
    self.h = 0.0001
    self.run_sa = False
    self.sa_count = 100
    self.sa_temp_start = 10.0
    self.sa_temp_end = 0.1
    self.sa_temp_reduction = 0.9
    self.parameters = numpy.zeros((10))
    self.parameters_upper = None
    self.parameters_lower = None
    self.function = None
    self.p_count = None
    self.lam_cutoff = 0.1
    self.lam = 0.1
  #  Load Data

  def load_file(self, file_name=None):
    if(file_name == None):
      return False
    # Init variable
    file_data = ""
    # Read it in line by line
    fh = open(file_name, "r")
    for file_row in fh:
      file_data = file_data + file_row.strip() + '\n'
    # Clean
    file_data = lma.clean(file_data)
    self.load_data(file_data)

  def load_data(self, file_data):  #
    lines = file_data.split('\n')
    data_list = []
    for line in lines:
      fields = line.split(',')
      if(len(fields) == 2):
        try:
          data_list.append([float(fields[0]), float(fields[1])])
        except:
          pass
    self.set_data_a(data_list)

  def set_data_a(self, data):
    self.data = numpy.zeros((len(data), 2))
    for row in range(len(data)):
      self.data[row,0] = float(data[row][0])
      self.data[row,1] = float(data[row][1])
    self.data_len = len(self.data)
    if(self.verbose):
      print("Data (a):")
      self.output_data()

  def set_data_b(self, data):
    self.data = numpy.zeros((len(data[0]), 2))
    for row in range(len(data[0])):
      self.data[row,0] = float(data[0][row])
      self.data[row,1] = float(data[1][row])
    self.data_len = len(self.data)
    if(self.verbose):
      print("Data (b):")
      self.output_data()

  def set_data_c(self, data):
    self.data = numpy.zeros((len(data), 2))
    for row in range(len(data)):
      self.data[row,0] = float(data[row,0])
      self.data[row,1] = float(data[row,1])
    self.data_len = len(self.data)
    if(self.verbose):
      print("Data (c):")
      self.output_data()

  def output_data(self):
    for row in range(self.data_len):
      print(self.data[row,0], self.data[row,1])
  #  Setters

  def set_threshold(self, convThreshold):
    self.conv_thr = conv_thr

  def set_fit(self, func, p):
    # Set function and parameter count
    self.function = func
    self.p_count = len(p)
    # Set parameters
    self.parameters = numpy.zeros((len(p)))
    for i in range(len(p)):
      self.parameters[i] = p[i]

  def set_sa(self, settings, pl=None, pu=None):
    #{"temp_start": 10.0, "temp_end": 1.0, "factor": 0.5, "count": 10}
    self.sa_count = settings['count']
    self.sa_temp_start = settings['temp_start']
    self.sa_temp_end = settings['temp_end']
    self.sa_temp_reduction = settings['factor']
    # Set parameter bounds
    self.parameters_lower = numpy.zeros((len(pl)))
    self.parameters_upper = numpy.zeros((len(pu)))
    for i in range(len(pl)):
      if(pl != None):
        self.parameters_lower[i] = pl[i]
    for i in range(len(pl)):
      if(pu != None):
        self.parameters_upper[i] = pu[i]
    if(pl != None and pu != None and self.p_count == len(pl) and self.p_count == len(pu)):
      self.run_sa = True
  #  Calc

  def calc(self):
    self.rss_start = self.calc_rss()
    p_input = numpy.copy(self.parameters)
    while(True):
      try:
        self.sa()
        self.outer_cycle()
        break
      except:
        self.parameters = numpy.copy(p_input)
        pass
    return self.parameters, self.calc_rss()

  def sa(self):
    if(self.run_sa == False):
      return 0
    p_opt = numpy.copy(self.parameters)
    rss_opt = self.calc_rss()
    print("Lower: ",self.parameters_lower)
    print("Opt:   ",p_opt)
    print("Upper: ",self.parameters_upper)
    temperature = self.sa_temp_start
    while(temperature > self.sa_temp_end):
      n = 0
      while(n<self.sa_count):
        p_best = numpy.copy(self.parameters)
        rss_best = self.calc_rss()
        # Vary
        self.parameters = self.parameters_lower[:] + (self.parameters_upper[:] - self.parameters_lower[:]) * numpy.random.rand(len(self.parameters_lower))
        rss = self.calc_rss()
        if(rss < rss_best):
          rss_best = rss
          if(rss_best < rss_opt):
            rss_opt = rss_best
            p_opt = numpy.copy(self.parameters)
        else:
          a = self.sa_acceptance(temperature, rss_best, rss)
          if(a > random.random()):
            rss_best = rss
          else:
            self.parameters = numpy.copy(p_best)
        # Increment
        n = n + 1
      # Reload optimum and cool
      self.parameters = numpy.copy(p_opt)
      temperature = temperature * self.sa_temp_reduction
    if(self.verbose):
      print(self.parameters, rss_opt)

  def sa_acceptance(self, temperature, best, new):
    return numpy.exp((best - new) / temperature)

  def outer_cycle(self):
    # (JTJ+Lambda*diag(JTJ)) P = (-1*JTR)
    # (H+Lambda*diag(H)) P = (-1*JTR)
    i = 0
    self.converged = False
    while(i < 100 and self.converged == False):
      i = i + 1
      self.make_residual()        # R
      self.make_jacobian()        # J
      self.make_jacobian_t()      # JT
      self.make_hessian()         # H ~ JTJ
      self.inner_cycle()
      if(self.calc_rss() < self.conv_thr):
        self.converged = True

  def inner_cycle(self):
    last_rss = self.calc_rss()
    for i in range(0,20):
      # Store last values
      p_last = numpy.copy(self.parameters)
      # Calculate matrices
      self.make_hessian()         # H ~ JTJ
      self.make_dampening()
      self.make_nJTR()            # -JTR
      self.dampen_hessian()
      self.update_parameters()
      rss = self.calc_rss()
      # Set parameters/rss
      if(rss>last_rss):
        self.parameters = numpy.copy(p_last)
        self.lam = self.lam * 1.5e0
      elif(rss == last_rss):
        self.lam = self.lam * 0.2e0
      else:
        last_rss = rss
        self.lam = self.lam * 0.2e0

  def make_residual(self):
    # Calculate residual
    self.r = self.function(self.parameters, self.data[:, 0]) - self.data[:, 1]

  def make_jacobian(self):
    self.J = numpy.zeros((self.data_len, self.p_count))
    for i in range(0, self.data_len):
      for j in range(0, self.p_count):
        # Reset parameters
        for k in range(self.p_count):
          p = numpy.copy(self.parameters)
        # Vary jth parameter
        p[j] = p[j] + self.h
        r = (self.function(p, self.data[i, 0]) - self.data[i, 1])
        self.J[i,j] = (r - self.r[i]) / self.h

  def make_jacobian_t(self):
    self.JT = numpy.transpose(self.J)

  def make_hessian(self):
    self.H = numpy.matmul(self.JT, self.J)

  def make_dampening(self):
    self.damp = numpy.identity(self.p_count)
    for i in range(0,self.p_count):
      self.damp[i,i] = self.lam * self.H[i,i]

  def make_nJTR(self):
    self.nJTR = -1 * numpy.matmul(self.JT, self.r)

  def dampen_hessian(self):
    for i in range(0,self.p_count):
      self.H[i,i] = self.H[i,i] + self.damp[i,i]
#l_cutoff

  def update_parameters(self):
    # A x = y
    x = numpy.linalg.solve(self.H, self.nJTR)
    p = numpy.copy(self.parameters)
    self.parameters[0:self.p_count] = p[0:self.p_count] + x[:]

  def calc_rss(self):
    return sum((self.function(self.parameters, self.data[:, 0]) - self.data[:, 1])**2)
# Static Functions

  @staticmethod
  def clean(str_in):
    str_out = ""
    l = len(str_in)
    for i in range(l):
      # Last, Next, This
      if(i == 0):
        last = None
      else:
        last = str_in[i-1]
      if(i < (l-1)):
        next = str_in[i+1]
      else:
        next = None
      char = str_in[i]
      # Check
      ok = True
      if(last == " " and char == " "):
        ok = False
      elif(last == "\n" and char == "\n"):
        ok = False
      elif(last == "\n" and char == " "):
        ok = False
      elif(char == " " and next == "\n"):
        ok = False
      # Add to string
      if(ok):
        str_out += char
    return str_out
# Example 1


# CLASS: FITTING
######################################

class fitting:

  @staticmethod
  def eos_guess(V, E):
    eos = {"E0": 0.0, "V0": 0.0, "B0": 0.0, "B0P": 0.0}
    # 2nd order polynomial fit
    poly = numpy.polyfit(V, E, 2)
    # Starting points
    eos['V0'] = (-1 * poly[1]) / (2 * poly[0])
    eos['E0'] = (poly[0] * eos['V0'] * eos['V0']) + (poly[1] * eos['V0']) + poly[2]
    eos['B0'] = 2.0 * poly[0] * eos['V0']
    eos['B0P'] = 2.0
    best_rss = fitting.rss(fitting.bm_calc, V, E, eos)
    eos_test = {}
    for i in range(10000):
      eos_test['V0'] = eos['V0'] + 0.0001 * (random.random() - 0.5)
      eos_test['E0'] = eos['E0'] + 0.0001 * (random.random() - 0.5)
      eos_test['B0'] = eos['B0'] + 0.0001 * (random.random() - 0.5)
      eos_test['B0P'] = eos['B0P'] + 0.0001 * (random.random() - 0.5)
      test_rss = fitting.rss(fitting.bm_calc, V, E, eos_test)
      if(test_rss < best_rss):
        eos['V0'] = eos_test['V0']
        eos['E0'] = eos_test['E0']
        eos['B0'] = eos_test['B0']
        eos['B0P'] = eos_test['B0P']
        best_rss = test_rss
    return eos

  @staticmethod
  def bm_calc(V, eos):
    V0 = eos['V0']
    E0 = eos['E0']
    B0 = eos['B0']
    B0P = eos['B0P']
    eta = (V/V0)**(1/3.0)
    return E0 + (9/16.0) * (B0 * V0) * ((eta*eta - 1)*(eta*eta - 1)) * (6.0 + B0P * (eta * eta - 1) - 4 * eta * eta )

  @staticmethod
  def rss(func, x_arr, y_arr, p):
    return sum((func(x_arr, p) - y_arr)**2)



# Program
########################################################

#    Processing PWscf input file
#
#
#
#
new_run = pwscf_eos()
#new_run.set_template("al.in")
new_run.run()
