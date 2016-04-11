__author__ = 'brycerich'

if __name__ == '__main__':
    deposit = 300
    interest = 1.006666666
    months = 12*13
    amount = 0
    while months >= 0:
        amount += deposit * interest ** months
        months -= 1
    amount = amount * interest ** (12*30)

    print("student A: ")
    print(amount)
    print("\n")

    deposit = 600
    interest = 1.006666666
    months = 12*30
    amount = 0
    while months >= 0:
        amount += deposit * interest ** months
        months -= 1

    print("student B: ")
    print(amount)
    print("\n")