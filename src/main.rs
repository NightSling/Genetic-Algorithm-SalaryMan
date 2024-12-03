use std::{fmt::Debug, i64, io::Write};

use genevo::{
    ga::genetic_algorithm,
    mutation::order::SwapOrderMutator,
    operator::prelude::{
        ElitistReinserter, RouletteWheelSelector,
    },
    prelude::{
        build_population, simulate, FitnessFunction, GenerationLimit, Population,
        SimResult, Simulation, SimulationBuilder,
    },
};
use utils::{
    Float32FitnessValue, OrderedSinglePointCrossBreeder, UniqueValueEncodedGnomeBuilder
};
mod utils;

// Attempting Salary Man Problem with Genevo crate

// Phenotype
#[derive(Clone, Debug, PartialEq)]
struct Problem {
    population: Population<Vec<usize>>, // The population of genomes or individuals.
    curr_gen: usize,                    // The current generation.
    towns: Vec<Town>,                   // The list of towns.
}

// Genotype
type Selection = Vec<usize>;

impl<'a> FitnessFunction<Selection, Float32FitnessValue> for &'a Problem {
    fn fitness_of(&self, a: &Selection) -> Float32FitnessValue {
        let mut dist: i64 = 0;

        // Calculate the total distance of the path.
        for i in 0..a.len() - 1 {
            let town1 = &self.towns[a[i] as usize];
            let town2 = &self.towns[a[i + 1] as usize];
            let x_diff = town1.x - town2.x;
            let y_diff = town1.y - town2.y;
            dist += (x_diff * x_diff + y_diff * y_diff) as i64; // Squared distance
        }
        Float32FitnessValue(1.0/(1.0 + dist as f32))

    }
    
    fn average(&self, a: &[Float32FitnessValue]) -> Float32FitnessValue {
        let sum: f32 = a.iter().map(|x| x.0).sum();
        Float32FitnessValue(sum / a.len() as f32)
    }
    
    fn highest_possible_fitness(&self) -> Float32FitnessValue {
        Float32FitnessValue(1.0)
    }
    
    fn lowest_possible_fitness(&self) -> Float32FitnessValue {
        0.0.into()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
struct Town {
    x: f32,
    y: f32,
}

// const TOWNS: [Town; 10] = [
//     Town { x: 93.0, y: 99.0 },
//     Town { x: 70.0, y: 44.0 },
//     Town { x: 5.0, y: 55.0 },
//     Town { x: 82.0, y: 66.0 },
//     Town { x: 42.0, y: 53.0 },
//     Town { x: 98.0, y: 53.0 },
//     Town { x: 79.0, y: 61.0 },
//     Town { x: 6.0, y: 32.0 },
//     Town { x: 58.0, y: 55.0 },
//     Town { x: 65.0, y: 61.0 },
// ];
// A more dynamic approach is used to generate random towns below

const POPULATION_SIZE: usize = 200;
const CROSSOVER_RATE: f64 = 0.85;
const MUTATION_RATE: f64 = 0.05;

fn main() {
    let mut fitness_over_generation: Vec<Float32FitnessValue> = Vec::new();
    
    let towns = generate_list_of_random_towns(1000);
    let population = build_population()
        .with_genome_builder(UniqueValueEncodedGnomeBuilder::new(
            towns.len(),
            0,
            towns.len() - 1,
        ))
        .of_size(POPULATION_SIZE)
        .uniform_at_random();

    let problem: Problem = Problem {
        population,
        curr_gen: 0,
        towns,
    };

    let mut sims = simulate(
        genetic_algorithm()
            .with_evaluation(&problem)
            .with_selection(RouletteWheelSelector::new(CROSSOVER_RATE, 2))
            .with_crossover(OrderedSinglePointCrossBreeder::new())
            .with_mutation(SwapOrderMutator::new(MUTATION_RATE))
            .with_reinsertion(ElitistReinserter::new(&problem, false, 0.85))
            .with_initial_population((&problem.population).clone())
            .build(),
    )
    .until(GenerationLimit::new(100))
    .build();

    loop {
        let result = sims.step();

        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;

                println!(
                    "step: generation: {}, average_fitness: {}, \
                     best fitness: {}, duration: {}, processing_time: {}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness,
                    step.duration,
                    step.processing_time
                );
                fitness_over_generation.push(evaluated_population.highest_fitness().clone());
            }
            Ok(SimResult::Final(step, processing_time, duration, reason)) => {
                let best_solution = step.result.best_solution;
                println!("{}", reason);
                println!(
                    "Final result after {}: generation: {}, \
                     best solution with fitness {} found in generation {}, processing_time: {}",
                    duration,
                    step.iteration,
                    best_solution.solution.fitness,
                    best_solution.generation,
                    processing_time,
                );
                println!("Best solution: {:?}", best_solution.solution);
                break;
            }
            Err(error) => {
                println!("Error: {:?}", error);
                break;
            }
        }
    }
    let mut file = std::fs::File::create("fitness_over_generation.csv").unwrap();
    for i in 0..fitness_over_generation.len() {
        file.write_all(fitness_over_generation[i].to_string().as_bytes()).unwrap();
        if i != fitness_over_generation.len() - 1 {
            file.write_all(b",").unwrap();
        }
    }
    file.write_all(b"\n").unwrap();
}

fn generate_list_of_random_towns(n: usize) -> Vec<Town> {
    (0..n).map(|_| generate_random_town()).collect()
}

fn generate_random_town() -> Town {
    Town {
        x: rand::random::<f32>() * 10000.0,
        y: rand::random::<f32>() * 10000.0,
    }
}
