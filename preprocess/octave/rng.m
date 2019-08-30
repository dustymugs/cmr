function rng(seed)
    % octave does not have seed
    randn('seed', seed);
    rand('seed', seed);
end
