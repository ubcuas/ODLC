from copy import deepcopy

class Tests:
    RECOGNIZED_LOC = "Code/Tests/recognized.txt"
    ACTUAL_LOC = "Code/Tests/actual.txt"

    REC_HIGH = 0
    REC_MODAL = 1

    def __init__(self):
        self.LIST_OF_ACTUAL = self.setup_list_of_actual()
        self.LIST_OF_RECOGNIZED = self.setup_list_of_recognized()

    def setup_list_of_actual(self):
        loa = []

        with open(self.ACTUAL_LOC) as file:
            for line in file:
                loa.append(str(line,)[0].lower())
            loa.pop(0)
        
        return loa
    
    def setup_list_of_recognized(self):
        lor = []

        with open(self.RECOGNIZED_LOC) as file:
            for line in file:
                #print(str(line,)[-2])
                #print(str(line,)[-5])
                #NOTE: CAREFUL WITH EMPTY LINE AT THE END
                lor.append([str(line,)[-5], str(line,)[-2]])
            lor.pop(0)
        
        return lor

    
    def highest_accuracy(self):
        #print("Apple" , end ="," )
        loa = deepcopy(self.LIST_OF_ACTUAL)
        lor = deepcopy(self.LIST_OF_RECOGNIZED)

        correct = []
        wrong_guess = []
        #not_guessed = []

        for i, letter in enumerate(lor):
            if letter[self.REC_HIGH] in loa:
                correct.append(letter[self.REC_HIGH])
                loa.pop(loa.index(letter[self.REC_HIGH]))
            else:
                wrong_guess.append(letter[self.REC_HIGH])
        
        print()
        print("Using Highest Confidence: ")
        print(bcolors.GREEN + " ".join(correct) + " " + bcolors.YELLOW + " ".join(loa) + bcolors.RED + " " + " ".join(wrong_guess) + bcolors.RESET)
        print(str(len(correct)) + "/" + str(len(self.LIST_OF_ACTUAL)) + "  (" + str(len(correct)/len(self.LIST_OF_ACTUAL)) + ")")
        print()

    # Could easily be combined with highest_accuracy, but future tests with color and stuff might want these functions separate
    def modal_accuracy(self):
        loa = deepcopy(self.LIST_OF_ACTUAL)
        lor = deepcopy(self.LIST_OF_RECOGNIZED)

        correct = []
        wrong_guess = []
        #not_guessed = []

        for i, letter in enumerate(lor):
            if letter[self.REC_MODAL] in loa:
                correct.append(letter[self.REC_MODAL])
                loa.pop(loa.index(letter[self.REC_MODAL]))
            else:
                wrong_guess.append(letter[self.REC_MODAL])
        
        print("Using Modal Confidence: ")
        print(bcolors.GREEN + " ".join(correct) + " " + bcolors.YELLOW + " ".join(loa) + bcolors.RED + " " + " ".join(wrong_guess) + bcolors.RESET)
        print(str(len(correct)) + "/" + str(len(self.LIST_OF_ACTUAL)) + "  (" + str(len(correct)/len(self.LIST_OF_ACTUAL)) + ")")
        print()


    """
    Tests when color and location are included:
    - Instead of only a letter in actual.txt, it will also include color and coordinates
    - So Recognized will then also have their estimated color and the given coordinates of that cropped image they chose
    - From this, instead of finding if they have an overlap in guesses, they will be compared by all three values 
        - Partial test points will be given with images with locations that are close enough 
                (with more partial iff at that location they also guess color or character)
    """


class bcolors:
    GREEN = '\033[92m' #GREEN
    YELLOW = '\033[93m' #YELLOW
    RED = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR
    """
    print(bcolors.GREEN + "File Saved Successfully!" + bcolors.RESET)
    print(bcolors.YELLOW + "Warning: Are you sure you want to continue?" + bcolors.RESET)
    print(bcolors.RED + "Unable to delete record." + bcolors.RESET)

    print(f"{bcolors.GREEN}File Saved Successfully!{bcolors.RESET}")
    print(f"{bcolors.YELLOW}Warning: Are you sure you want to continue?{bcolors.RESET}")
    print(f"{bcolors.RED}Unable to delete record.{bcolors.RESET}")
    """