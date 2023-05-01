import os
from jinja2 import Environment, select_autoescape, FileSystemLoader

CURRENT_PATH = os.path.abspath(__file__)


class ReportGenerator:
    """
    Class to generate reports in Markdown and PDF formats for OBS data.
    """
    def __init__(self, type='qc'):
        self._env = None
        self._jinja_env = None
        self._template_path = None

        if type == 'qc':
            self.TEMPLATE_NAME = 'qc_report.md'
        elif type == 'ops':
            self.TEMPLATE_NAME = 'ops_report.md'
        elif type == 'final':
            self.TEMPLATE_NAME = 'final_report.md'

    @property
    def env(self):
        if self._jinja_env is None:
            self._jinja_env = Environment(
                loader=FileSystemLoader(self.template_path),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
        if self._env is None:
            self._env = self._jinja_env.get_template(self.TEMPLATE_NAME)
        return self._env

    @property
    def template_path(self):
        if self._template_path is None:
            self._template_path = os.path.join(os.path.dirname(os.path.dirname(CURRENT_PATH)), 'templates')
        return self._template_path

    def write_report(self, input_variables, output_path=None):
        """
        Fill in template file with relevant info and write to `output_path`

        :param input_variables: dictionary of variables to fill in template
        :param output_path: full path to output file
        """
        if 'latString' not in input_variables:
            if 'latitude' in input_variables:
                if input_variables['latitude'] < 0:
                    input_variables['latString'] = '{0:.6f} S'.format(-input_variables['latitude'])
                else:
                    input_variables['latString'] = '{0:.6f} N'.format(input_variables['latitude'])
            else:
                input_variables['latString'] = 'none'
        if 'lonString' not in input_variables:
            if 'longitude' in input_variables:
                if input_variables['longitude'] < 0:
                    input_variables['lonString'] = '{0:.6f} W'.format(-input_variables['longitude'])
                else:
                    input_variables['lonString'] = '{0:.6f} E'.format(input_variables['longitude'])
            else:
                input_variables['lonString'] = 'none'

        if 'deployDate' not in input_variables:
            if 'deployed' in input_variables:
                input_variables['deployDate'] = input_variables['deployed'].strftime('%Y-%m-%d')
        if 'recoverDate' not in input_variables:
            if 'recovered' in input_variables:
                input_variables['recoverDate'] = input_variables['recovered'].strftime('%Y-%m-%d')

        if 'psdWindowLength' not in input_variables:
            if 'psdWindowSecs' in input_variables:
                if input_variables['psdWindowSecs'] < 3*60:     # 3 minutes
                    input_variables['psdWindowLength'] = '{0} second'.format(input_variables['psdWindowSecs'])
                elif input_variables['psdWindowSecs'] < 3*60*60:    # 3 hours
                    input_variables['psdWindowLength'] = '{0} minute'.format(input_variables['psdWindowSecs'] / 60)
                elif input_variables['psdWindowSecs'] < 3*60*60*24:    # 3 days
                    input_variables['psdWindowLength'] = '{0} hour'.format(input_variables['psdWindowSecs'] / 60 / 60)
                else:
                    input_variables['psdWindowLength'] = '{0} day'.format(input_variables['psdWindowSecs'] / 60 / 60 / 24)

        report_str = (self.env.render(
            **input_variables
        ))
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
        return output_path, report_str
