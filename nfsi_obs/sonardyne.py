from datetime import datetime
import re
import warnings


class SonardyneParseError(Exception):
    """
    Sonardyne parsing error.
    """


class SonardyneReader:
    """
    Parse Sonardyne Command Language communications
    """
    def __init__(self):
        """ Constructor """
        self._created = datetime.now()

    @staticmethod
    def parse(self, message):
        parsed = {}
        idx = 1
        if message[0] == '<':
            parsed['message_type'] = 'command'
        elif message[0] == '>':
            parsed['message_type'] = 'response'
        else:
            if re.match(r'\[.*\]', message[idx:]):
                parsed['message_type'] = 'data'
                idx += 1
                message = message[:-1]
            else:
                parsed['message_type'] = 'manual command'
                idx = 0

        if message[idx] == 'U':
            parsed['transceiver_UID'] = message[idx:idx+7]
            idx += 8    # Uhhhhhh,

        comm = re.match(r'([A-Za-z]+):(\d{4})?[,;]?', message[idx:])
        if comm is not None:
            parsed['command'] = comm.groups()[0].upper()
            parsed['address'] = comm.groups()[1]
            idx += len(comm[0])
        else:
            raise SonardyneParseError('Invalid message, skipping: {}'.format(message))

        parsed['message'] = message[idx:]

        # TODO: Interpret commands other than just MR
        if parsed['command'] == 'MR':
            # parse range information and stats
            if parsed['message_type'] == 'response':
                try:
                    stat_split = re.match(r'(.*)\[([A-Za-z0-9,;\-]+)\]', message[idx:])
                    ac_stats = stat_split.groups()[1].split(',')
                    for ac in ac_stats:
                        name = re.match(r'[A-Z]+', ac)
                        val = ac[len(name[0]):]
                        if len(val.split(';')) > 1:
                            # USBL response, multiple elements
                            parsed[name[0]] = [int(x) for x in val.split(';')]
                        else:
                            parsed[name[0]] = int(val)
                    rem = stat_split.groups()[0].split(';')
                    for r in rem:
                        info = re.match(r'([A-Z]+)([0-9e\-]+)', r)
                        parsed[info.groups()[0]] = float(info.groups()[1])
                except (TypeError, AttributeError):
                    warnings.warn('Invalid range measurement: {}'.format(message))

        return parsed
