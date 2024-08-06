import numpy as np
import math, os
import pickle


def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def watermark_str_to_numpy(watermark_str):
    result = [int(i) for i in watermark_str]
    return np.asarray(result)


def to_ascii(str_8_bits):
    power = 7
    res = 0
    for i in str_8_bits:
        res += int(i) * (2 ** power)
        power -= 1
    return res


def int_to_8_bit_binary_str(number):
    return '{0:08b}'.format(number)


def watermark_str_to_bitstr(watermark_str):
    binary_str = ""
    for letter in watermark_str:
        binary_str += int_to_8_bit_binary_str(ord(letter))
    return binary_str


def binary_literal_to_str(watermark_binary_literal):
    num_chunks = len(watermark_binary_literal) // 8

    return_str = ""
    for i_chunk in range(num_chunks):
        str_slice = watermark_binary_literal[8*i_chunk:8*(i_chunk+1)]
        ascii_code = int(to_ascii(str_slice))
        letter = chr(ascii_code)
        return_str += letter
    return return_str


if __name__ == "__main__":
    # ############### DEMO 1 #######################
    # # # === generate gt watermark in np array ===
    # # watermark_gt = np.random.binomial(1, 0.5, 32)  
    # # print("Watermark Numpy: ", watermark_gt)

    # # # === convert from np to str ===
    # # watermark_str = watermark_np_to_str(watermark_gt)
    # # print("Watermark String: ", watermark_str)

    # # # === convert from np to binary ===
    # # watermark_utf = watermark_str.encode("utf-8")
    # # print("Watermark machine code: ", watermark_utf)

    # # # === convert from binary to string ===
    # # watermark_recovered = watermark_utf.decode("utf-8")
    # # print("Recovered watermark: ", watermark_recovered)

    # # # === Further convert it into numpy arr ==
    # # watermark_final = watermark_str_to_numpy(watermark_recovered)
    # # print("Watermark Final: ", watermark_final)

    # # """
    # #     I guess for stegaStamp you need to use <watermark_utf> as the encoder input,

    # #     and need to convert the decoded message into <watermark_recovered>
    # # """


    # ########### DEMO 2 ##############
    # test_watermark_str = "asdf"   # Suppose you randomly draw a string with length 4. After uft-8 encoding, this will become a 32-bit length array.
    # print("\n Randomly drawn watermark: [{}]  \n".format(test_watermark_str))

    # test_watermark_binary_str = watermark_str_to_bitstr(test_watermark_str)
    # print("Converted Watermark (binary str): ", test_watermark_binary_str)
    # print("  Length: ", len(test_watermark_binary_str), "\n")

    # # Sanity check for binary conversion
    # str_recover = binary_literal_to_str(test_watermark_binary_str)
    # print("Sanity check for binary conversion: ", str_recover, "\n")
    
    # utf_code = test_watermark_str.encode("utf-8")
    # # Below is what the stegaStamp source code does: append it to 7-string length but we don't need the appended length since we only need 32 bits.
    # data = bytearray(test_watermark_str + ' '*(7-len(test_watermark_str)), 'utf-8')
    # print("Compare Binary code: ")
    # print("  UTF code: ", utf_code)
    # print("  Stega Stamp UTF code: ", data)  # You can see they are the same binary (within the first 4 strings)
    #                                        # You can further look into the diff. between .encode() and bytearray(), but it is irrelevant here.


    # # === What you need to do is that:
    # # 1) Given a watermarked img ---- I_w
    # # 2) Decode the img and get the decoded str ---  string_decoded = stegaStamp.decode(I_w)
    # # 3) Slice the first 4 characters:               watermark_str_decoded = string_decoded[:4]
    # # 3) Convert the string to binary lateral:       binary_lateral = watermark_str_to_bitstr(watermark_str_decoded)

    # ##  Test Load data ===
    # dir = os.path.join(
    #     ".", "Visualization", "711", "rivaGan", "dip",
    #     "vanila", "result.pkl"
    # )
    # with open(dir, 'rb') as f:
    #     data = pickle.load(f)
    
    # print(data.keys())
    # print(len(data["interm_recon"]))
    # print(data["iter_log"])

    # print(data["interm_recon"][5])

    # Count File numbers 
    watermarker = "rivaGan"
    dataset = "DiffusionDB"
    evader = "dip"
    arch = "vanila"
    result_dir = os.path.join(
        "Result-Interm", watermarker, dataset, evader, arch
    )
    file_list = [f for f in os.listdir(result_dir)]
    print("Number of files processed: ", len(file_list))
    print("Completed.")