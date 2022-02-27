import math
from decimal import Decimal

class Diffie_Hellman:
    def __init__(self):
        # This prime number is taken from 2048-bit MODP group described in RFC 3526
        # The reason for using this prime number is because of security purpose. 
        # More information can be found here: https://weakdh.org/imperfect-forward-secrecy-ccs15.pdf
        # self.prime = (1 << 2048) - (1 << 1984) - 1 + (1 << 64) * \
            # (math.floor(Decimal(1 << 1918) * Decimal(math.pi))  + 124476)
        #### To fasten the key generation process, I use 768-bit MODP group, which provides weaker security
        #### For more information, https://datatracker.ietf.org/doc/html/rfc7296#appendix-B.1
        self.prime = (1 << 768) - (1 << 704) - 1 + (1 << 64) * \
            (math.floor(Decimal(1 << 638) * Decimal(math.pi)) + 149686)
        # In terms of generator, I use 2, which follows the convention in https://datatracker.ietf.org/doc/html/rfc3526#page-3    
        self.generator = 2

    def generate_public_key(self, secret_key):
        # Since the generator is 2 so 2 ** secret_key is equivalent to 
        # shift bit 1 to the left secret_key # of times
        return pow(self.generator, secret_key, self.prime)
        #return self.f(self.generator, secret_key, self.prime)

    def compute_shared_secret_key(self, pub_key, secret_key):
        return pow(pub_key, secret_key, self.prime)
    

