from reedsolo import RSCodec

def repetition_encode(bits, n=3):
    """Repeat each bit n times for redundancy."""
    return [b for bit in bits for b in [bit]*n]

def repetition_decode(bits, n=3):
    """Majority vote decoding for repetition code."""
    chunks = [bits[i:i+n] for i in range(0, len(bits), n)]
    return [1 if sum(chunk) > n/2 else 0 for chunk in chunks]

def rs_encode(data_bytes, nsym=10):
    """Reed-Solomon encoding for bytes."""
    rsc = RSCodec(nsym)
    return bytes(rsc.encode(bytearray(data_bytes)))

def rs_decode(data_bytes, nsym=10):
    """Reed-Solomon decoding."""
    rsc = RSCodec(nsym)
    try:
        return bytes(rsc.decode(bytearray(data_bytes)))
    except:
        return None  # failed to recover
