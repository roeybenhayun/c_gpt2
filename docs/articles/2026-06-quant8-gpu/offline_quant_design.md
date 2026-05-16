-- Phase 1 - Offline Quantization --
1. quant would be W8A8, and data type would be BF16 (continue from what we previously did)
2. Pre-implamentation - offline quantization
2.1 Find the weights the would be quantized - those are the input to the GEMM
2.2.Plot the float32 values of one of the weight tensor - this would show the distribution of numbers 
2.2.1 show the amax, scale and the quantized result
2.3 Show few concrete example of mapping floa32 to quant8
2.4 Map which weights will be quantized offline and which not
2.5 Show the memory reduction before and after using quant for the weights of GEMM
3. Build the quant tool
3.1 Create a tools directory, and add the tool name under this directory. Something like offline_quant or quant or quantizer.
3.2 implement this in python and sandboxing using uv 
3.3 By default reads the weight downloaded from hugging face
3.4 This tool should be used for step2 by adding the following cli flags
3.4.1 --list/--show : this will show the list of the wights available for quant. This should be based on a config/mapping files that describes/maps weight (layer), if that layer is quantized
3.4.2 --distrib: it will ask the user to pick a weight and layer (if applicacble) from a list. The output would be a grpah with the fp32 distribuations, on the plot add the amax (also output it), and generate the scale tensor for this layer. 
3.4.3 output should be stored under tools/<toolname>/out/logs for logs, scale tensor (bin format) should be saved under tools/<toolename>/scales.
3.4.4 as part of the --distrib flag I would like also the plot the distrib loss when comming back from quant8 to bf16. I am not sure about the plot type. Offer different options as part of your planning. 
3.4.5 add a --stats flag. This would show the memory reduction (before and after)
3.4.6 add a --model flag. Could be either small/medium/large. Default is all the models
3.4.7 Basic run is just call the tool without any flag, it will creating the quantizer bin file for each model. use a quant schema with the following options:
"schema":"",
"scale_convention" : "",
"scale_dtype" : "",
"quantized_tensors" : ["",""],
"preserved_tensors" : ["",""],
"total_mem_before" : "",
"totle_mem_after" :""

I am not sure if to have the schema in a separate file or baked in the bin file. Present the pros and cons while you are reading this doc and planning. I am in favour of having a quant_config.json file.
3.4.8 add the end of the run ask the user if he would like to copy the files to main weights location under c_gpt2/weights. If yes, the python code would copy the files and also save them to out under the tool/<tool_name>
3.4.9 add another flag --valid. This would validate the results. This means it will go through all the quntized tensors and the associated scales and convert back to bf16 and measure the noise. Now here I would like to further consolut. Since the starting point should be fp32, and in the code I am casting it to bf16. So I am not sure how to measure the noise. Is it agains the original fp32 or bf16. 
In the valid phase you have to make sure to not change to shape of the tensors since the code is relying on it.
3.5 once this is validated. I would upload the files manually to hugging face, to the same place I store the weights. Once uploaded I will provide the url to download. Add download files to the setup script so now it would also downloade the quant files. 

-- Phase 1 - end of Offline Quantization --

