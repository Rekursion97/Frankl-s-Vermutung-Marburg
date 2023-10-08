import random
import numpy as np


def validate_assumption(values, eigen):
    return eigen / ((values[0] * 0.5) * values[1])


class Union_closed_Family:

    def __init__(self, max_number, amount_start_sets, max_size_start_sets):
        self.__max_number = max_number
        self.__amount_start_sets = amount_start_sets
        self.__max_size_start_sets = max_size_start_sets
        self.__family__ = set()
        self.__prime_sets__ = set()
        self.__create_union_closed_family(max_number, amount_start_sets, max_size_start_sets)

        self.__binary_matrix__ = None
        self.__prime_sets_binary_matrix__ = None
        self.__ATA__ = None
        self.__AAT__ = None
        self.__prime_sets_binary_matrix_ATA__ = None
        self.__prime_sets_binary_matrix_AAT__ = None
        self.__create_binary_matrices()

        # Meta information
        self.__amount_sets__ = 0
        self.__amount_elements__ = 0
        self.__occurrences_elements__ = {}
        self.__occurrences_elements_prime_sets__ = {}
        self.__amount_elements_total__ = 0
        self.__frequency_elements__ = {}
        self.__frequency_elements_prime_sets__ = {}
        self.__create_meta_information()

    @classmethod
    def __create_binary_matrix(cls, sets):
        max_value = -1
        for currSet in sets:
            for number in currSet:
                if number > max_value:
                    max_value = number
        matrix = np.zeros((len(sets), max_value + 1))
        i = 0
        for currSet in sets:
            for number in currSet:
                matrix[i, number] = 1
            i = i + 1
        return matrix

    @classmethod
    def __create_ATA(cls, A):
        return A.T @ A

    @classmethod
    def __create_AAT(cls, A):
        return A @ A.T

    @classmethod
    def __compute_max_eigenvalue(cls, A):
        return np.max(np.linalg.eigvals(A))

    @classmethod
    def __compute_frequency_and_perecentage(cls, A, amount_sets):
        dims = np.shape(A)
        count_elements = {}
        frequency_elements = {}
        for i in range(0, dims[1]):
            curr_counter = 0
            for j in range(0, dims[0]):
                curr_counter = curr_counter + A[j, i]
            count_elements[i] = curr_counter
            frequency_elements[i] = (curr_counter / amount_sets) * 100
        return [count_elements, frequency_elements]

    def __create_union_closed_family(self, m, amount_start_sets, max_size_start_sets):
        current_max = -1
        for i in range(0, amount_start_sets):
            current_list = []
            for element in range(0, random.randrange(1, max_size_start_sets, 1)):
                next_element = random.randrange(0, m, 1)
                #if next_element - current_max > 1:
                #    next_element = current_max + 1
                #    current_max = next_element
                current_list.append(next_element)
            self.__family__.add(frozenset(current_list))
        while self.__family__.__contains__(frozenset()):
            self.__family__.remove(frozenset())

        test = self.__family__
        # Create prime sets. Prime sets are the sets which can only be reproduced by itself
        for current_set in self.__family__:
            is_prime = True
            for set1 in self.__family__:
                for set2 in self.__family__:
                    if set1 != set2 and set1 != current_set and set2 != current_set:
                        if set1.union(set2) == current_set:
                            is_prime = False
                            break
                if not is_prime:
                    break
            if is_prime:
                self.__prime_sets__.add(current_set)

        # Fill family with all missing sets
        while True:
            sets_to_add = []
            for set1 in self.__family__:
                for set2 in self.__family__:
                    if set1 != set2:
                        union = set1.union(set2)
                        if not self.__family__.__contains__(union):
                            sets_to_add.append(union)
            if len(sets_to_add) > 0:
                for currSet in sets_to_add:
                    self.__family__.add(currSet)
            else:
                break

    def __create_binary_matrices(self):
        self.__binary_matrix__ = self.__create_binary_matrix(self.__family__)
        self.__ATA__ = self.__create_ATA(self.__binary_matrix__)
        self.__AAT__ = self.__create_AAT(self.__binary_matrix__)

        self.__prime_sets_binary_matrix__ = self.__create_binary_matrix(self.__prime_sets__)
        self.__prime_sets_binary_matrix_ATA__ = self.__create_ATA(self.__prime_sets_binary_matrix__)
        self.__prime_sets_binary_matrix_AAT__ = self.__create_AAT(self.__prime_sets_binary_matrix__)

    def compute_max_eigenvalue_ATA(self):
        return self.__compute_max_eigenvalue(self.__ATA__)

    def compute_max_eigenvalue_AAT(self):
        return self.__compute_max_eigenvalue(self.__AAT__)

    def compute_max_eigenvalue_prime_ATA(self):
        return self.__compute_max_eigenvalue(self.__prime_sets_binary_matrix_ATA__)

    def compute_max_eigenvalue_prime_AAT(self):
        return self.__compute_max_eigenvalue(self.__prime_sets_binary_matrix_AAT__)

    def most_common_element(self):
        dims = np.shape(self.__binary_matrix__)
        max_amount = 0
        element = 0
        for i in range(0, dims[1]):
            curr_counter = 0
            for j in range(0, dims[0]):
                curr_counter = curr_counter + self.__binary_matrix__[j, i]
            if curr_counter > max_amount:
                max_amount = curr_counter
                element = i
        return [element, max_amount]

    def __create_meta_information(self):
        dims = np.shape(self.__binary_matrix__)
        self.__amount_sets__ = dims[0]
        self.__amount_elements__ = dims[1]

        # count_elements: Dictionary containing: element -> Frequency of element.
        # frequency_elements: Dictionary containing element -> percentage occurrence in all sets
        res_non_prime = self.__compute_frequency_and_perecentage(self.__binary_matrix__, self.__amount_sets__)
        self.__occurrences_elements__ = res_non_prime[0]
        self.__frequency_elements__ = res_non_prime[1]

        res_prime = self.__compute_frequency_and_perecentage(self.__prime_sets_binary_matrix__,
                                                             np.shape(self.__prime_sets_binary_matrix__)[0])
        self.__occurrences_elements_prime_sets__ = res_prime[0]
        self.__frequency_elements_prime_sets__ = res_prime[1]
        # Sum of cardinality of every set
        total = 0
        for current_set in self.__family__:
            total = total + len(current_set)
        self.__amount_elements_total__ = total

    def amount_sets(self):
        return self.__amount_sets__

    def amount_different_elements(self):
        return self.__amount_elements__

    def frequency_elements(self):
        return self.__occurrences_elements__

    def percentage_elements(self):
        return self.__frequency_elements__

    def amounts_elements_total(self):
        return self.__amount_elements_total__

    def meta_information(self):
        print("The family: \n")
        print(self.__family__)
        print("\n\n")
        print("Amount of sets: \n")
        print(self.__amount_sets__)
        print("\n\n")
        print("Amount different elements: \n")
        print(self.__amount_elements__)
        print("\n\n")
        print("Sum of cardinality all sets: \n")
        print(self.__amount_elements_total__)
        print("\n\n")
        print("Frequency elements in sets: \n")
        print(self.__occurrences_elements__)
        print("\n\n")
        print("Percentage of occurrence element in a set: \n")
        print(self.__frequency_elements__)
        print("\n\n")
        print("The corresponding binary matrix B: \n")
        print(self.__binary_matrix__)
        print("\n\n")
        print("Solution of transpose(B)*B: \n")
        print(self.__ATA__)
        print("\n\n")
        print("With biggest eigenvalue: \n")
        print(self.compute_max_eigenvalue_ATA())
        print("Solution of B*transpose(B): \n\n")
        print(self.__AAT__)
        print("\n\n")
        print("With biggest eigenvalue: \n")
        print(self.compute_max_eigenvalue_AAT())
        print("\n\n")
        print("The prime family: \n")
        print(self.__prime_sets__)
        print("\n\n")
        print("Frequency elements in prime sets: \n")
        print(self.__occurrences_elements_prime_sets__)
        print("\n\n")
        print("Percentage of occurrence element in a prime set: \n")
        print(self.__frequency_elements_prime_sets__)
        print("\n\n")
        print("The corresponding prime binary matrix PB: \n")
        print(self.__prime_sets_binary_matrix__)
        print("\n\n")
        print("Solution of transpose(PB)*PB: \n")
        print(self.__prime_sets_binary_matrix_ATA__)
        print("\n\n")
        print("With biggest eigenvalue: \n")
        print(self.compute_max_eigenvalue_prime_ATA())
        print("\n\n")
        print("Solution of PB*transpose(PB): \n")
        print(self.__prime_sets_binary_matrix_AAT__)
        print("\n\n")
        print("With biggest eigenvalue: \n")
        print(self.compute_max_eigenvalue_prime_AAT())
        print("\n\n")
