use std::fmt::{Debug, Display};

use genevo::{
    genetic::{AsScalar, Children, Parents},
    operator::{CrossoverOp, GeneticOperator},
    prelude::{Fitness, GenomeBuilder, Genotype},
};
use rand::Rng;

/// A genome builder that creates a genome with unique values.
/// The genome is a vector of unsigned bytes. The values are in the range [min_index, max_index].
/// The genome is created by randomly selecting values from the range [min_index, max_index] and
/// removing them from the available values.
///
/// # Example
/// ```
/// use genevo::prelude::GenomeBuilder;
/// use utils::UniqueValueEncodedGnomeBuilder;
///
/// let genome_builder = UniqueValueEncodedGnomeBuilder::new(10, 0, 9);
/// let mut rng = rand::thread_rng();
/// let genome = genome_builder.build_genome(0, &mut rng);
/// ```
///
///
pub struct UniqueValueEncodedGnomeBuilder {
    size: usize,
    min_index: usize,
    max_index: usize,
}

impl UniqueValueEncodedGnomeBuilder {
    /// Creates a new genome builder that creates a genome with unique values.
    /// The genome is a vector of unsigned bytes. The values are in the range [min_index, max_index].
    ///
    /// # Arguments
    /// * `size` - The size of the genome.
    /// * `min_index` - The inclusive minimum value of the genome
    /// * `max_index` - The inclusive maximum value of the genome.
    pub fn new(size: usize, min_index: usize, max_index: usize) -> Self {
        Self {
            size,
            min_index,
            max_index,
        }
    }
}

impl GenomeBuilder<Vec<usize>> for UniqueValueEncodedGnomeBuilder {
    fn build_genome<R>(&self, _: usize, rng: &mut R) -> Vec<usize>
    where
        R: rand::Rng + Sized,
    {
        let mut genome: Vec<usize> = Vec::with_capacity(self.size);
        let mut available_values: Vec<usize> = (self.min_index..=self.max_index).collect();
        for _ in 0..self.size {
            let random_index = rng.gen_range(0..available_values.len());
            genome.push(available_values.remove(random_index));
        }
        genome
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OrderedSinglePointCrossBreeder;

impl OrderedSinglePointCrossBreeder {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticOperator for OrderedSinglePointCrossBreeder {
    fn name() -> String {
        "Ordered-Single-Point-Crossover".to_string()
    }
}

impl<G> CrossoverOp<G> for OrderedSinglePointCrossBreeder
where
    G: Genotype + OrderSinglePointCrossOver,
{
    fn crossover<R>(
        &self,
        parents: genevo::genetic::Parents<G>,
        rng: &mut R,
    ) -> genevo::genetic::Children<G>
    where
        R: rand::Rng + Sized,
    {
        G::crossover(parents, rng)
    }
}

pub trait OrderSinglePointCrossOver: Genotype {
    type Dna;

    fn crossover<R>(parents: Parents<Self>, rng: &mut R) -> Children<Self>
    where
        R: Rng + Sized;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Float32FitnessValue(pub f32);

impl AsScalar for Float32FitnessValue {
    fn as_scalar(&self) -> f64 {
        self.0 as f64
    }
}

impl From<f32> for Float32FitnessValue {
    fn from(value: f32) -> Self {
        Float32FitnessValue(value)
    }
}

impl Into<f32> for Float32FitnessValue {
    fn into(self) -> f32 {
        self.0
    }
}

impl Display for Float32FitnessValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // display with maximum precision, aka as many digits as possible
        write!(f, "{:.}", self.0)
    }
}

impl Ord for Float32FitnessValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for Float32FitnessValue {
    fn assert_receiver_is_total_eq(&self) {
        // This method is never called; it exists only to assert that `Eq` is implemented.
    }
}


impl Fitness for Float32FitnessValue {
    fn zero() -> Self {
        Float32FitnessValue(0.0)
    }

    fn abs_diff(&self, other: &Self) -> Self {
        Float32FitnessValue((self.0 - other.0).abs())
    }
}

impl OrderSinglePointCrossOver for Vec<usize> {
    type Dna = usize;

    fn crossover<R>(parents: Parents<Self>, rng: &mut R) -> Children<Self>
    where
        R: Rng + Sized,
    {
        let gnome_size = parents[0].len();
        assert!(
            parents.len() == 2,
            "This crossover expects exactly two parents."
        );

        let mut child1 = vec![0; gnome_size];
        let mut child2 = vec![0; gnome_size];

        let crossover_point = rng.gen_range(0..gnome_size);
        let parent1_head = &parents[0][0..crossover_point];
        let parent2_head = &parents[1][0..crossover_point];

        child1[0..crossover_point].copy_from_slice(parent1_head);
        child2[0..crossover_point].copy_from_slice(parent2_head);

        let child1_set = parent1_head
            .iter()
            .collect::<std::collections::HashSet<_>>();
        let child2_set = parent2_head
            .iter()
            .collect::<std::collections::HashSet<_>>();

        for val in parents[0].iter() {
            if !child2_set.contains(val) {
                let index = child2.iter().position(|x| *x == 0).unwrap();
                child2[index] = *val;
            }
        }

        for val in parents[1].iter() {
            if !child1_set.contains(val) {
                let index = child1.iter().position(|x| *x == 0).unwrap();
                child1[index] = *val;
            }
        }

        vec![child1, child2]
    }
}
