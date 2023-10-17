from pb import create_population, init_run, _evaluate_fitness

p = create_population(size=2, problem_description= "Solve the math word problem, giving your answer as an arabic numeral.")

p = init_run(p)

for x in p.units:
    print("\n\n\n" + "#"*60)
    print(x.P)

_evaluate_fitness(p)
