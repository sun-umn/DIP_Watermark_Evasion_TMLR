import numpy as np

def watermark_np_to_str(watermark_np):
    """
        Convert a watermark in np format into a str to display.
    """
    return "".join([str(i) for i in watermark_np.tolist()])


def watermark_str_to_numpy(watermark_str):
    result = [int(i) for i in watermark_str]
    return np.asarray(result)


if __name__ == "__main__":

    # === generate gt watermark in np array ===
    watermark_gt = np.random.binomial(1, 0.5, 32)  
    print("Watermark Numpy: ", watermark_gt)

    # === convert from np to str ===
    watermark_str = watermark_np_to_str(watermark_gt)
    print("Watermark String: ", watermark_str)

    # === convert from np to binary ===
    watermark_utf = watermark_str.encode("utf-8")
    print("Watermark machine code: ", watermark_utf)

    # === convert from binary to string ===
    watermark_recovered = watermark_utf.decode("utf-8")
    print("Recovered watermark: ", watermark_recovered)

    # === Further convert it into numpy arr ==
    watermark_final = watermark_str_to_numpy(watermark_recovered)
    print("Watermark Final: ", watermark_final)

    """
        I guess for stegaStamp you need to use <watermark_utf> as the encoder input,

        and need to convert the decoded message into <watermark_recovered>
    """