from monodromy.static.interference import check_main_xx_theorem, regenerate_xx_solution_polytopes

print("Checking main global theorem.")
check_main_xx_theorem()
print("Done.")

print("Checking main local theorem.")
regenerate_xx_solution_polytopes()
print("Done.")