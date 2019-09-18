function init_cmr()
    [this_path, junk, junk] = fileparts(mfilename('fullpath'));

    addpath(fullfile(this_path, '..', 'utils'));
    addpath(fullfile(this_path, '..', 'sfm'));
    addpath(fullfile(this_path, '..', 'quaternions'));

    if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
        graphics_toolkit('gnuplot')
        setenv('GNUTERM', 'x11')

        %debug_on_warning(1);
        debug_on_error(1);

        page_output_immediately(1);
        page_screen_output(0);

        addpath(fullfile(this_path, '..', 'octave'));
    end
end
