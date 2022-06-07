import numpy as np
def run(xaxis, data, **kwargs):
    derivative = np.gradient(data, xaxis)
    derivative = (derivative - derivative.mean())/derivative.std()
    script_outputs =  {"data":derivative}
    script_outputs["xaxis"] = xaxis
    script_outputs["xaxis_title"]='q'
    script_outputs["data_title"]= 'derivative'
                   