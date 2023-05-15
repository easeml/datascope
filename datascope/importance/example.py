from datascope.utility import Units

x = Units(units=5, candidates=2)  # units:int (units=range(int)+frozen); units:list (frozen); units:None (free form)

x(0) == 0 & x(1) == 0 | x(2) == 1
