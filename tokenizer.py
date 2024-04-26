from collections import defaultdict
import random

def get_pair_frequency(corpus):
    pair_freq = defaultdict(int)
    for token in corpus:
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freq[pair] += 1
    return pair_freq

def merge_pair(pair, corpus):
    new_corpus = []
    for token in corpus:
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(pair[0] + '_' + pair[1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_corpus.append(new_token)
    return new_corpus

def build_bpe_tokenizer(corpus, max_tokens):
    # Initialize the vocabulary with individual characters
    vocabulary = {char: i for i, char in enumerate(set(corpus))}
    
    # Split the corpus into individual characters
    tokenized_corpus = [list(token) for token in corpus.split()]
    
    # Iteratively merge the most frequent pairs
    while len(vocabulary) < max_tokens:
        pair_freq = get_pair_frequency(tokenized_corpus)
        if not pair_freq:
            break
        
        most_freq_pair = max(pair_freq, key=pair_freq.get)
        tokenized_corpus = merge_pair(most_freq_pair, tokenized_corpus)
        
        new_token = most_freq_pair[0] + '_' + most_freq_pair[1]
        vocabulary[new_token] = len(vocabulary)
    
    return vocabulary

def tokenize_string(string, tokenizer):
    tokenized = []
    chars = list(string)
    i = 0
    while i < len(chars):
        # Find the longest matching token
        longest_match = None
        for j in range(i + 1, len(chars) + 1):
            token = '_'.join(chars[i:j])
            if token in tokenizer:
                longest_match = token
        if longest_match:
            tokenized.append(tokenizer[longest_match])
            i += len(longest_match.split('_'))
        else:
            # Handle unknown tokens
            if chars[i] in tokenizer:
                tokenized.append(tokenizer[chars[i]])
            else:
                tokenized.append(tokenizer['<UNK>'])
            i += 1
    return tokenized

def tokenize_string_random(string, tokenizer, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    
    tokenized = []
    chars = list(string)
    i = 0
    while i < len(chars):
        # Find all matching tokens
        matching_tokens = []
        for j in range(i + 1, len(chars) + 1):
            token = '_'.join(chars[i:j])
            if token in tokenizer:
                matching_tokens.append(token)
        
        if matching_tokens:
            # Randomly select a token from the matching tokens
            selected_token = random.choice(matching_tokens)
            tokenized.append(tokenizer[selected_token])
            i += len(selected_token.split('_'))
        else:
            # Handle unknown tokens
            if chars[i] in tokenizer:
                tokenized.append(tokenizer[chars[i]])
            else:
                tokenized.append(tokenizer['<UNK>'])
            i += 1
    
    return tokenized

TOKENIZER = {'u': 0, 'ñ': 1, 'z': 2, 'A': 3, '6': 4, 'n': 5, 'R': 6, 'G': 7, 'r': 8, 'e': 9, 'm': 10, 'k': 11, 'j': 12, '$': 13, 's': 14, 'q': 15, 'ā': 16, 'K': 17, '8': 18, 'U': 19, "'": 20, 'Z': 21, '7': 22, 'O': 23, 'L': 24, 'é': 25, 'o': 26, 'B': 27, '1': 28, '¡': 29, '-': 30, 'C': 31, 'F': 32, '!': 33, '&': 34, 'T': 35, 't': 36, 'y': 37, 'x': 38, 'P': 39, 'J': 40, '£': 41, 'l': 42, 'V': 43, '…': 44, 'c': 45, 'p': 46, 'H': 47, 'h': 48, 'â': 49, '5': 50, 'v': 51, 'I': 52, 'ó': 53, 'w': 54, '—': 55, '.': 56, 'f': 57, 'd': 58, '0': 59, 'è': 60, 'Q': 61, '�': 62, 'N': 63, '3': 64, '"': 65, '%': 66, 'W': 67, 'ū': 68, 'á': 69, 'Y': 70, 'E': 71, 'M': 72, 'X': 73, 'g': 74, '?': 75, 'D': 76, ',': 77, ' ': 78, 'S': 79, '4': 80, '2': 81, 'i': 82, 'a': 83, '9': 84, 'b': 85, '\n': 86, 'h_e': 87, 't_h_e': 88, 'i_n': 89, 'a_n': 90, 'r_e': 91, 'o_n': 92, 'o_u': 93, 'e_r': 94, 't_h': 95, 'a_t': 96, 't_o': 97, 'e_d': 98, 'a_n_d': 99, 'e_n': 100, 'a_s': 101, 'i_s': 102, 'o_f': 103, 'i_n_g': 104, 'o_r': 105, 'a_r': 106, 'e_s': 107, 'i_t': 108, 'a_l': 109, 'l_e': 110, 'a_d': 111, 's_e': 112, 's_t': 113, 'o_m': 114, 'b_e': 115, 'h_i': 116, 'o_w': 117, 'l_y': 118, 'c_h': 119, 'w_a_s': 120, 'l_l': 121, 'l_d': 122, 'w_i': 123, 'v_e': 124, 'c_e': 125, 't_h_a_t': 126, 'm_e': 127, 'i_d': 128, 'g_h': 129, 'n_o': 130, 'e_n_t': 131, 'u_t': 132, 'w_e': 133, 'i_o_n': 134, 'h_i_s': 135, 'v_e_r': 136, 'l_i': 137, 'a_y': 138, 'f_o_r': 139, 'i_r': 140, 'y_o_u': 141, 'w_i_t_h': 142, 'r_i': 143, 'T_h_e': 144, 'h_a_d': 145, 'a_c': 146, 'r_o': 147, 'n_d': 148, 'h_a': 149, 'h_e_r': 150, 't_e_r': 151, 'a_l_l': 152, 's_u': 153, 'g_h_t': 154, 's_.': 155, 'n_o_t': 156, 'w_h': 157, 't_h_e_r': 158, 'o_u_l_d': 159, 'l_o': 160, 'n_e': 161, 'f_e': 162, 's_a': 163, 'i_c': 164, 'k_e': 165, 's_,': 166, 'd_e': 167, 'a_b': 168, 'm_o': 169, 'h_i_m': 170, 'o_n_e': 171, 'a_g': 172, 's_o': 173, 'u_r': 174, 'a_i_n': 175, 'p_e': 176, 'o_u_t': 177, 'o_m_e': 178, 'r_e_d': 179, 'a_m': 180, 't_i': 181, 's_h': 182, 'p_o': 183, 'u_n': 184, 's_h_e': 185, 'w_h_i': 186, "'_s": 187, 'c_o_n': 188, 'm_y': 189, 'f_r': 190, 'u_p': 191, 'e_t': 192, 'o_d': 193, 'b_y': 194, 'h_a_v_e': 195, 'm_a_n': 196, 'c_a': 197, 'i_l': 198, 'c_t': 199, 'w_e_r_e': 200, 't_e': 201, 'w_h_i_c_h': 202, 'g_o': 203, 'u_l': 204, 'b_u_t': 205, 'p_e_r': 206, 'u_s': 207, 'a_r_d': 208, 'e_s_s': 209, 'w_h_e': 210, 'H_e': 211, 'f_r_o_m': 212, 'r_e_s': 213, 'o_u_r': 214, 'e_x': 215, 'q_u': 216, 'i_f': 217, 't_r': 218, 'a_r_e': 219, 't_h_e_y': 220, 'n_o_w': 221, 'd_o': 222, 'a_t_i_o_n': 223, 't_h_i_s': 224, 'f_o': 225, 's_a_i_d': 226, 'i_n_d': 227, 'w_h_o': 228, 'a_n_t': 229, 'i_m': 230, 'a_p': 231, 't_h_i_n_g': 232, 'p_l': 233, 'v_e_r_y': 234, 'c_o_m': 235, 't_h_e_m': 236, 'e_s_t': 237, 'o_k': 238, 'o_w_n': 239, 's_e_l': 240, 'a_r_t': 241, 'w_a_y': 242, 'u_s_t': 243, 'o_n_g': 244, 'o_u_s': 245, 'g_e': 246, 'a_t_e': 247, 't_h_e_i_r': 248, 'e_n_d': 249, 'w_o_u_l_d': 250, 'A_n_d': 251, 'a_s_t': 252, 'c_o': 253, 'o_u_n_d': 254, 'i_g_h_t': 255, 's_o_m_e': 256, 'w_o_r': 257, 'i_l_l': 258, 'I_t': 259, 'e_d_.': 260, 'o_p': 261, 'o_u_g_h': 262, 'p_r': 263, 'y_.': 264, 'o_t_h_e_r': 265, 'f_a': 266, 'g_r': 267, 'p_r_o': 268, 'i_s_t': 269, 's_e_e': 270, 'b_e_e_n': 271, 'a_k': 272, 'B_u_t': 273, 's_e_l_f': 274, 't_h_e_r_e': 275, 'b_o': 276, 'i_t_t': 277, "'_t": 278, 'e_s_,': 279, 'v_e_n': 280, 'c_k': 281, 'y_,': 282, 'a_c_k': 283, 'a_f': 284, 'e_d_,': 285, 'w_h_e_n': 286, 'a_n_y': 287, 'c_o_u_l_d': 288, 'm_e_n_t': 289, 'w_i_l_l': 290, 'u_r_e': 291, 'g_r_e': 292, 'i_g': 293, 'f_o_r_e': 294, 'a_k_e': 295, 'a_b_l_e': 296, 'e_s_.': 297, 'm_o_r_e': 298, 'i_n_t_o': 299, 'r_a': 300, 'h_o': 301, 'w_h_a_t': 302, 'w_o': 303, 'k_n_o_w': 304, 'c_a_n': 305, 'u_m': 306, 'i_v_e': 307, 'i_d_e': 308, 'o_u_g_h_t': 309, 'h_e_d': 310, 'c_i': 311, 'M_r': 312, 'd_i_d': 313, 'y_o_u_r': 314, 'a_n_c_e': 315, 'e_p': 316, 'l_a': 317, 'd_a_y': 318, 'o_u_n': 319, 'd_i_s': 320, 'l_i_k_e': 321, 'i_t_e': 322, 'h_a_t': 323, 'u_n_d': 324, 'p_t': 325, 'b_l': 326, 't_h_a_n': 327, 'i_t_t_l_e': 328, 'p_o_s': 329, 'a_d_e': 330, 'i_s_h': 331, 'S_h_e': 332, 'i_n_e': 333, 'o_v_e_r': 334, 'u_r_n': 335, 'a_s_s': 336, 't_i_m_e': 337, 'a_b_o_u_t': 338, 'v_e_d': 339, 'l_i_t_t_l_e': 340, 'e_r_s': 341, 'f_u_l': 342, 'e_n_c_e': 343, 'p_p': 344, 's_p': 345, 'b_l_e': 346, 'h_a_s': 347, 'o_r_t': 348, 'l_o_o_k': 349, 'o_l': 350, 'm_e_d': 351, 't_h_e_n': 352, 'e_a_r': 353, 'a_l_l_y': 354, 'e_l': 355, 'p_r_e_s': 356, 'b_r': 357, 's_t_r': 358, 'i_t_s': 359, 'p_r_e': 360, 'a_g_e': 361, 'c_a_m_e': 362, 'v_i_n_g': 363, 'a_g_a_i_n': 364, 'r_e_n': 365, 'h_a_n_d': 366, 't_o_r': 367, 'W_e': 368, 'u_p_o_n': 369, 'i_t_y': 370, 'g_l': 371, 't_a_i_n': 372, 'i_e': 373, 'a_c_t': 374, 'p_l_e': 375, 'i_t_.': 376, 'l_o_n_g': 377, 'g_r_e_a_t': 378, 'i_n_k': 379, 'i_n_g_.': 380, 's_a_y': 381, 'o_n_l_y': 382, 'c_o_m_e': 383, 'a_f_t_e_r': 384, 't_u_r_n': 385, 'a_n_s': 386, 'u_s_e': 387, 'm_a_d_e': 388, 'i_c_e': 389, 'c_l': 390, 's_h_o_u_l_d': 391, 'm_e_n': 392, 'd_o_w_n': 393, 'o_l_d': 394, 'm_u': 395, 'i_n_g_,': 396, 'n_e_w': 397, 'm_o_s_t': 398, 'r_e_e': 399, 'c_r': 400, 'r_e_a_d': 401, 'a_n_g': 402, 'h_o_w': 403, 'a_c_e': 404, 'e_v_e_r': 405, 'l_y_.': 406, 't_h_r': 407, 'o_r_d': 408, 'o_t': 409, 'o_f_f': 410, 'i_r_e': 411, 'i_n_t': 412, 'u_e': 413, 'b_e_f_o_r_e': 414, 'Y_o_u': 415, 'w_a_r_d': 416, 'w_h_e_r_e': 417, 'o_b': 418, 'p_a_r_t': 419, 'f_l': 420, 's_i': 421, 'a_r_m': 422, 't_w_o': 423, 'b_e_r': 424, 'o_n_d': 425, 'T_h_e_y': 426, 'e_v_e_r_y': 427, 'a_v_e': 428, 'e_y': 429, 'g_o_o_d': 430, 's_u_c_h': 431, 'w_e_l_l': 432, 'b_a_c_k': 433, 'f_i_r': 434, 'c_h_i': 435, 'M_r_.': 436, 'm_u_c_h': 437, 'k_e_d': 438, 'p_a': 439, 's_p_e': 440, 'I_n': 441, 'n_e_v_e_r': 442, 'l_y_,': 443, 'a_c_h': 444, 'o_s_e': 445, 'e_v_e_n': 446, 'l_i_g_h_t': 447, 'm_u_s_t': 448, 's_o_n': 449, 'e_r_.': 450, 'd_e_s': 451, 'o_u_s_e': 452, 'a_t_e_d': 453, 'g_e_t': 454, 'u_n_d_e_r': 455, 'h_i_m_.': 456, 'h_e_a_d': 457, 's_t_o': 458, 's_e_r': 459, "n_'_t": 460, 'u_d': 461, 'o_r_n': 462, 'r_i_g_h_t': 463, 'v_e_s': 464, 'm_a_r': 465, "I_'": 466, 't_h_o_u_g_h_t': 467, 'j_u_s_t': 468, 'f_t': 469, 't_h_e_s_e': 470, 'l_e_t': 471, 'e_r_,': 472, 't_h_i_n_k': 473, 'T_h_e_r_e': 474, 'l_e_s_s': 475, 'b_e_t': 476, 'm_i_g_h_t': 477, 'c_r_e': 478, 'k_i_n_g': 479, 'w_e_n_t': 480, 'c_o_u_n': 481, 'e_f': 482, 'h_e_n': 483, 'c_a_r': 484, 'l_l_o_w': 485, 'f_i_r_s_t': 486, 'h_i_m_s_e_l_f': 487, 'h_e_r_e': 488, 'a_n_c': 489, 'r_y': 490, "o_n_'_t": 491, 'm_a_y': 492, 't_h_r_o_u_g_h': 493, 'b_e_g': 494, 'i_s_e': 495, 'l_e_d': 496, 't_t': 497, 'a_t_t': 498, 'i_o_u_s': 499, 'c_l_e': 500, 'r_e_s_t': 501, 's_i_d_e': 502, 'c_o_u_r': 503, 'h_o_u_s_e': 504, 'c_a_l_l': 505, 'e_m': 506, 'i_c_k': 507, 'i_o_n_.': 508, 't_r_e': 509, 'c_l_o': 510, 'a_r_y': 511, 'm_o_n': 512, 'p_s': 513, 'l_i_f_e': 514, 's_m': 515, 'l_t': 516, 'n_i_g_h_t': 517, 'i_s_s': 518, 't_o_o': 519, 'm_e_.': 520, 'i_g_n': 521, 'a_w_a_y': 522, 'r_i_e_d': 523, 'w_o_r_d': 524, 'm_a_k_e': 525, 'i_o_n_s': 526, 't_e_l_l': 527, 'w_o_m': 528, 'v_o': 529, 'w_a_t': 530, 'i_t_,': 531, 'n_g': 532, 'c_h_e': 533, 't_h_o_u_g_h': 534, 'w_i_t_h_o_u_t': 535, 'f_o_u_n_d': 536, 'l_a_n_d': 537, 'i_e_s': 538, 'l_a_s_t': 539, 'c_e_r': 540, 'b_e_i_n_g': 541, 'y_o_u_n_g': 542, 'r_o_o_m': 543, 'A_s': 544, 'p_a_s_s': 545, 'm_i_n': 546, 'c_o_n_t': 547, 'T_h_e_n': 548, 't_h_o_s_e': 549, 'F_o_r': 550, 'r_e_p': 551, 'o_p_l_e': 552, 'b_r_o': 553, 'n_e_s_s': 554, 't_i_m': 555, 'm_e_r': 556, 'f_o_r_t': 557, 'h_a_p_p': 558, 'a_t_h': 559, 'T_h_a_t': 560, 'c_c': 561, 'j_e': 562, 'y_e_a_r': 563, 'm_a_t': 564, 'g_u': 565, 'b_u': 566, 'm_a_n_y': 567, 'W_h_e_n': 568, 'N_o': 569, 'T_h_i_s': 570, 'c_a_u_s_e': 571, 'a_s_e': 572, 'W_h_a_t': 573, 'S_o': 574, 'm_e_,': 575, 's_t_i_l_l': 576, 'f_a_c_e': 577, 'l_i_e': 578, 'p_e_o_p_l_e': 579, 'a_i_r': 580, 'a_n_k': 581, 's_t_a_n_d': 582, 'i_z': 583, 'a_p_p_e': 584, 'p_u_t': 585, 'n_a': 586, 'w_h_i_l_e': 587, 'w_a_y_s': 588, 's_a_i_d_,': 589, 'f_r_i': 590, 'h_y': 591, 'c_h_i_l_d': 592, 'd_e_n': 593, 'w_a_n_t': 594, 'n_o_t_h_i_n_g': 595, 's_a_w': 596, 't_a_k_e': 597, 's_e_d': 598, 's_u_r': 599, 't_.': 600, 'f_f': 601, 'p_l_a_c_e': 602, 'l_e_f_t': 603, 'c_o_n_s': 604, 'g_i_r': 605, 'a_r_k': 606, 'i_l_y': 607, 'e_n_e_d': 608, 'f_e_c_t': 609, 'M_r_s_.': 610, 'j_o': 611, 'u_l_l': 612, 'm_i_n_d': 613, 'I_f': 614, 't_o_o_k': 615, 'w_o_r_k': 616, 't_e_d': 617, 'b_o_d': 618, 'g_i': 619, 'l_o_w': 620, 'n_a_t': 621, 'g_i_r_l': 622, 'a_n_g_e': 623, 'e_v': 624, 'f_a_t_h_e_r': 625, 'h_o_r': 626, 'h_e_a_r_t': 627, 'i_n_c_e': 628, 'h_i_m_,': 629, 'i_o_n_,': 630, 'c_o_m_p': 631, 'i_s_h_e_d': 632, 'o_n_c_e': 633, 't_e_n': 634, 'a_k_i_n_g': 635, 't_h_e_m_.': 636, "d_o_n_'_t": 637, 'a_,': 638, 's_a_m_e': 639, 'm_o_m': 640, 't_u_r_n_e_d': 641, 'f_o_r_m': 642, 'f_r_i_e_n_d': 643, 'd_o_o_r': 644, 'v_a_l': 645, 't_o_l_d': 646, 'a_u': 647, 's_h_a_l_l': 648, 'h_e_a_r_d': 649, 'm_o_m_e_n_t': 650, 't_a_l': 651, 'g_e_n': 652, 'a_n_o_t_h_e_r': 653, 'o_n_.': 654, 'l_o_v_e': 655, 'i_a_n': 656, 'b_r_e': 657, 's_h_i': 658, 'a_l_w_a_y_s': 659, 'i_n_t_e_r': 660, 'h_e_r_.': 661, 't_y': 662, 's_e_e_m_e_d': 663, 'w_a': 664, 's_c': 665, 'p_r_i': 666, 'g_i_v_e': 667, 'i_m_p': 668, 'o_n_,': 669, 'a_d_d': 670, 'p_o_i_n_t': 671, 'e_y_e_s': 672, 'd_i_f': 673, 'l_o_o_k_e_d': 674, 'j_e_c_t': 675, 'd_i_s_t': 676, 'w_i_n_d': 677, 'y_e_t': 678, 'u_b': 679, 'f_a_r': 680, 'w_r': 681, 'p_e_d': 682, 'r_o_w': 683, 'c_h_o': 684, 'c_e_s_s': 685, 'm_o_t_h_e_r': 686, 'v_i': 687, 'b_i_t': 688, 'A_t': 689, 'l_e_s': 690, 'd_i': 691, 'p_r_e_s_e_n_t': 692, 't_r_i': 693, 'p_e_c_t': 694, 's_s': 695, 's_a_t': 696, 't_e_m': 697, 'w_o_r_l_d': 698, 's_u_b': 699, 'C_h': 700, 'a_t_i_o_n_.': 701, 'r_e_g': 702, 'f_e_e_l': 703, 'l_l_,': 704, 's_e_n': 705, 'c_o_n_d': 706, 'm_e_m': 707, 'f_i_n_d': 708, 'p_a_r': 709, 'u_g': 710, 'l_i_c': 711, 'v_i_s': 712, 'c_h_e_d': 713, 'k_i_n_d': 714, 'T_h': 715, 'c_e_r_t_a_i_n': 716, 'w_n': 717, 'p_e_r_s_o_n': 718, 'h_o_m_e': 719, 's_o_m_e_t_h_i_n_g': 720, 'h_e_l': 721, 's_o_o_n': 722, 'h_,': 723, 'a_r_e_d': 724, 'n_e_a_r': 725, 't_i_n_g': 726, 'n_o_r': 727, 'b_e_c_a_u_s_e': 728, 't_h_r_e_e': 729, 'g_o_t': 730, 'a_.': 731, 's_t_e': 732, 'u_c_k': 733, 'a_u_t': 734, 'w_o_m_a_n': 735, 'g_o_i_n_g': 736, 'u_s_e_d': 737, 'h_a_l': 738, 's_e_t': 739, 'c_o_l': 740, 'H_i_s': 741, 'a_s_k': 742, 'k_n_e_w': 743, 'c_e_d': 744, 'c_r_i': 745, 'f_e_w': 746, 's_u_r_e': 747, 'c_o_m_m': 748, 'J_o': 749, 'a_g_a_i_n_s_t': 750, 'i_t_h': 751, 'p_l_e_a_s': 752, 'i_t_i_o_n': 753, 'a_v': 754, 'p_o_w': 755, 'f_i_n': 756, 'c_e_n_t': 757, 's_h_a': 758, 'r_e_s_s': 759, 'f_o_l_l_o_w': 760, 'o_n_e_.': 761, 's_i_l': 762, 'b_e_g_a_n': 763, 'o_p_e_n': 764, 's_o_l': 765, 'b_e_t_t_e_r': 766, '0_0': 767, 's_e_e_n': 768, 'w_a_l': 769, 'a_s_t_e_r': 770, 'e_n_o_u_g_h': 771, 'l_e_c_t': 772, 'b_e_l_i_e': 773, 't_r_y': 774, 'c_a_r_e': 775, 'T_o': 776, 'o_c_c': 777, 'm_a': 778, 'e_,': 779, 'p_o_s_s_i': 780, 'u_l_a_r': 781, 'q_u_i': 782, 'g_e_d': 783, 'c_o_u_r_s_e': 784, 's_t_r_e': 785, 's_y': 786, 'g_e_t_h_e_r': 787, 'd_r': 788, 'c_e_p_t': 789, 'm_o_r_n': 790, 't_o_g_e_t_h_e_r': 791, 'e_a_c_h': 792, 'i_n_n': 793, "I_'_m": 794, 'm_a_i_n': 795, 'f_e_l_t': 796, 'f_e_r_e_n': 797, 'e_.': 798, 't_,': 799, 'i_c_a_l': 800, 'F_r': 801, 'A_n': 802, 'e_n_g': 803, 'M_y': 804, 's_i_t': 805, 't_h_,': 806, 'p_u_r': 807, 'a_d_v': 808, 'w_e_e_n': 809, 'c_a_l_l_e_d': 810, 'w_h_o_l_e': 811, 'y_o_u_.': 812, 'a_i_l': 813, 'f_i': 814, 'p_o_s_e': 815, 'l_a_t': 816, 'h_o_u_r': 817, 'i_d_e_n_t': 818, 's_e_r_v': 819, 'b_a_r': 820, 'l_a_y': 821, 'M_a_r': 822, 'b_e_a_u_t': 823, 's_p_o': 824, 'h_a_r_d': 825, 's_h_o_w': 826, 't_h_.': 827, 'O_n': 828, 'b_e_t_w_e_e_n': 829, 'h_a_v_i_n_g': 830, 'i_n_g_s': 831, 'f_a_l_l': 832, 'b_e_d': 833, 'h_e_r_s_e_l_f': 834, 'f_a_i_r': 835, 'e_s_s_.': 836, 'r_e_a_d_y': 837, 'b_o_d_y': 838, 'B_e': 839, 'f_a_m': 840, 'g_e_n_t': 841, 'c_u': 842, 't_o_w_a_r_d': 843, 'a_k_e_n': 844, 't_i_l': 845, 'q_u_e_s_t': 846, 'i_n_t_e': 847, 's_h_o': 848, 'p_r_e_s_s': 849, 'E_n': 850, 'w_a_r': 851, 't_s': 852, 'm_e_a_n': 853, 's_e_n_t': 854, 'g_e_n_e_r': 855, 'h_a_l_f': 856, 's_k': 857, 'i_t_h_e_r': 858, 'a_s_k_e_d': 859, 'a_t_h_e_r': 860, 'a_t_i_o_n_s': 861, 'l_a_r': 862, 'e_r_e_d': 863, 'm_y_s_e_l_f': 864, 'o_u_d': 865, 'v_o_i_c_e': 866, 'M_i_s_s': 867, 'm_a_t_t_e_r': 868, 'd_r_e': 869, 'p_l_a_y': 870, 'a_w': 871, 'a_l_s_o': 872, 'f_a_c_t': 873, 'r_i_s': 874, 'c_e_i': 875, 'i_t_e_d': 876, 'i_t_y_.': 877, 'n_e_x': 878, 't_h_i_n_g_s': 879, 'q_u_i_t_e': 880, 's_m_a_l_l': 881, 'a_m_o_n_g': 882, 's_l_e': 883, 'i_a_l': 884, 'y_o_u_,': 885, 'g_a_v_e': 886, 'l_i_n_g': 887, 'e_r_i_n_g': 888, 'l_a_d': 889, 'c_h_a_r': 890, 'H_o_w': 891, 'a_s_o_n': 892, 'b_o_y': 893, 's_e_e_m': 894, 'd_d_e_n': 895, 'c_o_n_t_i_n': 896, 'w_h_i_t_e': 897, 'c_l_a': 898, 'h_e_r_,': 899, 'u_a_l': 900, 'h_o_l_d': 901, 'r_u': 902, 't_i_m_e_s': 903, 'E_n_g_l': 904, 'a_l_m_o_s_t': 905, 'a_t_i_n_g': 906, 'o_n_e_,': 907, 'd_o_e_s': 908, 'r_e_p_l_i': 909, 'a_n_y_t_h_i_n_g': 910, 'u_n_t_i_l': 911, 'w_o_o_d': 912, "'_r_e": 913, 'w_h_o_m': 914, 'r_o_u_n_d': 915, 'a_r_o_u_n_d': 916, 'u_c_t': 917, 'f_e_e_t': 918, 'N_o_w': 919, 'h_e_a_r': 920, 'n_e_e_d': 921, 'f_u_l_l': 922, 'a_t_i_o_n_,': 923, 'h_a_p_s': 924, 'v_e_r_s': 925, 'l_e_g': 926, 'h_i_g_h': 927, 'i_n_s_t': 928, 'w_o_n_d': 929, 'l_e_.': 930, 's_t_o_r': 931, 'W_e_l_l_,': 932, 'h_u': 933, 'p_o_s_s_i_b_l_e': 934, 'i_n_d_e': 935, 'f_r_e': 936, 'i_s_t_e_r': 937, 'e_v_e_r_,': 938, 'H_e_r': 939, 'e_k': 940, 'f_i_c': 941, 'k_e_e_p': 942, '._.': 943, 'c_o_r': 944, 'a_c_c': 945, 'p_l_a_i_n': 946, 'u_r_i_n_g': 947, 'd_a_r_k': 948, 's_o_u': 949, 'a_m_p': 950, 'b_r_o_u_g_h_t': 951, 'd_o_u': 952, 'i_e_n_t': 953, 'v_e_r_e_d': 954, 's_t_o_o_d': 955, 'a_l_o_n_g': 956, 'y_e_a_r_s': 957, 'p_o_o_r': 958, 'b_e_s_t': 959, 's_u_n': 960, 's_i_n_g': 961, 'i_n_s': 962, 'p_a_l': 963, 'l_i_m': 964, 's_t_e_p': 965, 'p_p_e_d': 966, 't_h_e_m_,': 967, 'g_g': 968, 'b_o_t_h': 969, 's_u_m': 970, 'w_i_s_h': 971, 'h_n': 972, 'c_a_l': 973, 'e_s_s_,': 974, 'h_i_n_d': 975, 's_t_e_r': 976, 'U_n': 977, 's_u_p': 978, 'Y_e_s_,': 979, 'A_l_l': 980, 'J_o_h_n': 981, 't_a_k_e_n': 982, 'i_n_t_e_r_e_s_t': 983, 'm_i_s': 984, 'g_r_o_u_n_d': 985, 'b_l_a_c_k': 986, 'g_a': 987, 'r_u_n': 988, 'm_o_n_e': 989, 'i_e_d': 990, 'e_n_d_e_d': 991, 'r_e_l': 992, 'r_o_s_s': 993, 'g_h_t_e_r': 994, 'b_e_l_i_e_v_e': 995, 'd_o_n_e': 996, 'a_c_h_e_d': 997, 'c_u_t': 998, 'p_o_w_e_r': 999, '<UNK>':1000}
INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}


if __name__ == '__main__':
    """
    corpus = ""

    with open('transcriptions.txt', 'r') as f:
        for line in f:
            if len(line.split(":")) > 2:
                print(line)
            corpus += line.split(":")[1] + " "
        
    print('done reading')
    max_tokens = 1000

    tokenizer = build_bpe_tokenizer(corpus, max_tokens)
    print(tokenizer, len(tokenizer))


    for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!:;\"\'":
        print(f"{char}: {tokenizer[char]}")
    """

    string = "Hey there! How are you doing today?"

    # Sample different tokenizations
    for i in range(5):
        tokenized_string = tokenize_string_random(string, TOKENIZER, random_state=i)
        print(f"Tokenization {i+1}: {tokenized_string}")
        for token in tokenized_string:
            print(f"{token}: {INV_TOKENIZER[token]}")
