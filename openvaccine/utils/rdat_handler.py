from collections import defaultdict


def split(s, delims=None):
    if ',' in delims:
        delims = delims.split(',')
    process = False
    for d in delims:
        if d in s:
            process = True
            break

    if process:
        for d in delims[1:]:
            s = s.replace(d, delims[0])
        return [x for x in s.split(delims[0]) if x]
    return [s]


class RDATSection(object):
    def __init__(self, attr_list=[], attr_str=[]):
        for attr in attr_list:
            setattr(self, attr, [])
        for attr in attr_str:
            setattr(self, attr, '')


class RDATFile(object):
    def __init__(self):
        self.constructs = defaultdict(list)

        self.values = defaultdict(list)
        self.errors = defaultdict(list)
        self.traces = defaultdict(list)
        self.reads = defaultdict(list)

        self.comments = ''
        self.annotations = defaultdict(list)
        self.data_types = defaultdict(list)
        self.mutpos = defaultdict(list)
        self.xsels = defaultdict(list)

        self.filename = None
        self.version = None
        self.loaded = False

    def _append_new_data_section(self, this_construct):
        d = RDATSection(['seqpos', 'values', 'errors', 'trace', 'reads', 'xsel'])
        d.annotations = {}
        self.constructs[this_construct].data.append(d)

    def _parse_data_block(self, line, key, start_idx=0):
        if not isinstance(key, list):
            key = [key]
        for k in key:
            attheader = k + ':' if ':' in line else k
            line = line.replace(attheader, '')

        fields = split(line.strip('\n ,'), delims='\t, ')
        data_idx = int(fields[0]) - 1 if start_idx else None
        data = [float(x) if ':' not in x else float(x[:x.find(':')]) for x in fields[start_idx:]]
        return (data, data_idx)

    def _parse_annotations(self, s):
        d = {}
        if self.version == 0.1:
            token = ';'
            s = s.split(',')
        else:
            token = ':'

        for item in s:
            if item:
                pair = item.split(token)
                if pair[0].strip() in d:
                    d[pair[0].strip()].append(':'.join(pair[1:]))
                else:
                    d[pair[0].strip()] = [':'.join(pair[1:])]
        return d

    def load(self, file):
        self.filename = file.name

        # only used for self.version == 0.1:
        current_section = 'general'
        fill_data_types = False
        # data_dict = {}

        lines = file.readlines()
        for line in lines:
            line = line.strip()

            if line.startswith('VERSION:'):
                self.version = float(line.replace('VERSION:', ''))
                continue
            elif line.startswith('RDAT_VERSION') or line.startswith('VERSION'):
                self.version = float(line.replace('RDAT_VERSION', '').replace('VERSION', ''))
                continue

            if self.version >= 0.2 and self.version < 0.4:
                if line.startswith('COMMENT'):
                    parsed_line = line
                    for sep in ' \t':
                        parsed_line = parsed_line.replace('COMMENTS' + sep, '').replace('COMMENT' + sep, '')
                    self.comments += parsed_line + '\n'

                elif line.startswith('ANNOTATION') and not line.startswith('ANNOTATION_DATA'):
                    self.annotations = self._parse_annotations(split(line.replace('ANNOTATION', ''), delims='\t'))

                elif 'CONSTRUCT' in line or line.startswith('NAME'):
                    if 'CONSTRUCT' in line:
                        line = file.readline().strip()  # Advance to 'NAME' line.

                    this_construct = line.replace('NAME', '').strip()
                    data_idx = -1
                    self.constructs[this_construct] = RDATSection(['seqpos', 'data', 'xsel'], ['sequence', 'structure'])
                    self.constructs[this_construct].name = this_construct
                    self.constructs[this_construct].annotations = {}
                    self.constructs[this_construct].structures = defaultdict(str)
                    self.constructs[this_construct].sequences = defaultdict(str)

                elif line.startswith('SEQUENCE'):
                    attheader = 'SEQUENCE:' if ':' in line else 'SEQUENCE'
                    line = line.replace(attheader, '')
                    if len(line.split()) > 1:
                        seqidx, seq = line.strip().split()
                        self.constructs[this_construct].sequences[int(seqidx)] = seq.strip()
                        self.constructs[this_construct].sequence = seq.strip()
                    else:
                        seq = line
                        self.constructs[this_construct].sequence = seq.strip()
                        self.constructs[this_construct].sequences[0] = seq.strip()

                elif line.startswith('STRUCTURE'):
                    attheader = 'STRUCTURE:' if ':' in line else 'STRUCTURE'
                    line = line.replace(attheader, '')
                    if len(line.split()) > 1:
                        structidx, struct = line.strip().split()
                        self.constructs[this_construct].structures[int(structidx)] = struct.strip()
                        self.constructs[this_construct].structure = struct.strip()
                    else:
                        struct = line
                        self.constructs[this_construct].structure = struct.strip()
                        self.constructs[this_construct].structures[0] = struct.strip()

                elif line.startswith('OFFSET'):
                    self.constructs[this_construct].offset = int(line.replace('OFFSET', ''))

                elif line.startswith('DATA_TYPE'):
                    self.data_types[this_construct] = split(line.replace('DATA_TYPE', '').strip(), delims='\t')

                elif line.startswith('SEQPOS'):
                    seqpos_tmp = split(line.replace('SEQPOS', '').strip(), delims='\t, ')
                    if self.version >= 0.32:
                        self.constructs[this_construct].seqpos = [int(x[1:]) for x in seqpos_tmp]
                    else:
                        self.constructs[this_construct].seqpos = [int(x) for x in seqpos_tmp]

                elif line.startswith('MUTPOS'):
                    self.mutpos[this_construct] = [x.strip() for x in split(line.replace('MUTPOS', '').strip(), delims='\t')]

                elif line.startswith('ANNOTATION_DATA'):
                    fields = split(line.replace('ANNOTATION_DATA:', '').replace('ANNOTATION_DATA ', '').strip(), delims='\t')
                    if len(fields) < 2:
                        fields = split(fields[0], delims=' ')
                    data_idx = int(fields[0]) - 1
                    annotations = self._parse_annotations(fields[1:])
                    for l in range(data_idx - len(self.constructs[this_construct].data) + 1):
                        self._append_new_data_section(this_construct)
                    self.constructs[this_construct].data[data_idx].annotations = annotations
                    if 'mutation' in annotations:
                        try:
                            if len(self.mutpos[this_construct]) > 0:
                                self.mutpos[this_construct][-1] = int(annotations['mutation'][0][1:-1])
                            else:
                                self.mutpos[this_construct].append(int(annotations['mutation'][0][1:-1]))
                        except ValueError:
                            pass

                elif line.startswith('AREA_PEAK') or line.startswith('REACTIVITY:'):
                    (peaks, data_idx) = self._parse_data_block(line, ['AREA_PEAK', 'REACTIVITY'], 1)
                    if (data_idx >= len(self.constructs[this_construct].data)):
                        self._append_new_data_section(this_construct)
                    self.constructs[this_construct].data[data_idx].values = peaks
                    self.values[this_construct].append(self.constructs[this_construct].data[data_idx].values)

                elif line.startswith('AREA_PEAK_ERROR') or line.startswith('REACTIVITY_ERROR:'):
                    (errors, data_idx) = self._parse_data_block(line, ['AREA_PEAK_ERROR', 'REACTIVITY_ERROR'], 1)
                    self.constructs[this_construct].data[data_idx].errors = errors
                    self.errors[this_construct].append(self.constructs[this_construct].data[data_idx].errors)

                elif line.startswith('TRACE'):
                    (trace, data_idx) = self._parse_data_block(line, 'TRACE', 1)
                    if data_idx < len(self.constructs[this_construct].data):
                        self.constructs[this_construct].data[data_idx].trace = trace
                        self.traces[this_construct].append(self.constructs[this_construct].data[data_idx].trace)

                elif line.startswith('READS'):
                    (reads, data_idx) = self._parse_data_block(line, 'READS', 1)
                    if data_idx < len(self.constructs[this_construct].data):
                        self.constructs[this_construct].data[data_idx].reads = reads
                        self.reads[this_construct].append(self.constructs[this_construct].data[data_idx].reads)

                elif line.startswith('XSEL_REFINE'):
                    (xsel, data_idx) = self._parse_data_block(line, 'XSEL_REFINE', 1)
                    self.constructs[this_construct].data[data_idx].xsel = xsel
                    self.xsels[this_construct].append(self.constructs[this_construct].data[data_idx].xsel)

                elif line.startswith('XSEL'):
                    (xsel, data_idx) = self._parse_data_block(line, 'XSEL', 0)
                    self.constructs[this_construct].xsel = xsel

                else:
                    if line.strip():
                        raise AttributeError('Invalid section: ' + line)
            else:
                raise ValueError('Wrong version number %s' % self.version)

            if self.version == 0.1 and fill_data_types:
                self.data_types[this_construct] = [self.data_types[this_construct][0]] * len(self.values[this_construct])

        if self.version >= 0.2:
            self.comments = self.comments[:-1]
        self.loaded = True
