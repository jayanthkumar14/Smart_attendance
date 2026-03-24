import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from backend.database.models import db
from backend.routes.students import students_bp
from backend.routes.attendance import attendance_bp

def create_app(config=None):
    app = Flask(__name__,
                static_folder='frontend/static',
                template_folder='frontend/templates')

    # ── Config ────────────────────────────────────────────────────────
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    app.config['SECRET_KEY']                    = os.getenv('SECRET_KEY', 'change-me-in-production')
    app.config['SQLALCHEMY_DATABASE_URI']       = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "attendance.db")}'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH']            = 16 * 1024 * 1024  # 16 MB upload cap

    if config:
        app.config.update(config)

    # ── Extensions ────────────────────────────────────────────────────
    CORS(app, resources={r'/api/*': {'origins': '*'}})
    db.init_app(app)

    # ── Blueprints ────────────────────────────────────────────────────
    app.register_blueprint(students_bp)
    app.register_blueprint(attendance_bp)

    # ── DB init ───────────────────────────────────────────────────────
    with app.app_context():
        db.create_all()
        _seed_subjects()

    # ── Frontend catch-all ────────────────────────────────────────────
    @app.route('/')
    @app.route('/<path:path>')
    def serve_frontend(path='index.html'):
        template_dir = os.path.join(BASE_DIR, 'frontend', 'templates')
        if path and os.path.exists(os.path.join(template_dir, path)):
            return send_from_directory(template_dir, path)
        return send_from_directory(template_dir, 'index.html')

    return app


def _seed_subjects():
    """Insert default subjects if the table is empty."""
    from backend.database.models import Subject
    if Subject.query.count() == 0:
        defaults = [
            ('Mathematics',   'MATH101', 'Science'),
            ('Physics',       'PHY101',  'Science'),
            ('Computer Sci.', 'CS101',   'Engineering'),
            ('Chemistry',     'CHEM101', 'Science'),
            ('English',       'ENG101',  'Humanities'),
        ]
        for name, code, dept in defaults:
            db.session.add(Subject(name=name, code=code, department=dept))
        db.session.commit()


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
