from html.parser import HTMLParser

class WorkspaceMarkup(HTMLParser):
    def __init__(self): super().__init__(); self.sections=[];self.elements={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.elements[attrs['id']]=(tag,attrs)
        if tag=='details' and 'data-workspace-section' in attrs:self.sections.append(attrs)

def test_initial_workspace_only_upload_is_open():
    from transformer_app import app
    markup=WorkspaceMarkup();markup.feed(app.test_client().get('/').get_data(as_text=True))
    assert len(markup.sections)==6
    assert [section['id'] for section in markup.sections if 'open' in section]==['upload-section']
    assert 'sidebar' not in markup.elements and 'sidebar-scrim' not in markup.elements
    assert 'prediction-placeholder' not in markup.elements
    assert 'hidden' in markup.elements['prediction-result'][1]
    for id in ['prediction-section','monitoring-section','explanation-section','report-section']:
        assert markup.elements[id][1]['aria-disabled']=='true'

def test_research_page_has_no_sidebar_either():
    from transformer_app import app
    markup=WorkspaceMarkup();markup.feed(app.test_client().get('/performance').get_data(as_text=True))
    assert 'sidebar' not in markup.elements
    assert 'model-metrics' in markup.elements and 'research-content' in markup.elements
