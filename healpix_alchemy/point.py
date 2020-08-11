"""Spatial indexing for astronomical point coordinates."""
import numpy as np
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.schema import Column, Index
from sqlalchemy.sql import and_, between
from sqlalchemy.types import Float

from .util import InheritTableArgs

__all__ = ('Point',)


def get_x_for_ra_dec(ra, dec):
    return np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))


def get_y_for_ra_dec(ra, dec):
    return np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))


def get_z_for_ra_dec(ra, dec):
    return np.sin(np.deg2rad(dec))


def get_x_for_context(context):
    params = context.get_current_parameters()
    return get_x_for_ra_dec(params['ra'], params['dec'])


def get_y_for_context(context):
    params = context.get_current_parameters()
    return get_y_for_ra_dec(params['ra'], params['dec'])


def get_z_for_context(context):
    params = context.get_current_parameters()
    return get_z_for_ra_dec(params['ra'], params['dec'])


class Point(InheritTableArgs):
    """Mixin class to add a point to a an SQLAlchemy declarative model."""

    def __init__(self, *args, ra=None, dec=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ra = ra
        self.dec = dec
        self.x = get_x_for_ra_dec(ra, dec)
        self.y = get_y_for_ra_dec(ra, dec)
        self.z = get_z_for_ra_dec(ra, dec)

    ra = Column(Float)
    dec = Column(Float)
    x = Column(Float, default=get_x_for_context, onupdate=get_x_for_context)
    y = Column(Float, default=get_y_for_context, onupdate=get_y_for_context)
    z = Column(Float, default=get_z_for_context, onupdate=get_z_for_context)

    @hybrid_property
    def cartesian(self):
        """Convert to Cartesian coordinates.

        Returns
        -------
        x, y, z : float
            A tuple of the x, y, and z coordinates.

        """
        return (self.x, self.y, self.z)

    @hybrid_method
    def within(self, other, radius):
        """Test if this point is within a given radius of another point.

        Parameters
        ----------
        other : Point
            The other point.
        radius : float
            The match radius in degrees.

        Returns
        -------
        bool

        """
        sin_radius = np.sin(np.deg2rad(radius))
        cos_radius = np.cos(np.deg2rad(radius))
        carts = (obj.cartesian for obj in (self, other))
        terms = ((between(lhs, rhs - 2 * sin_radius, rhs + 2 * sin_radius),
                  lhs * rhs) for lhs, rhs in zip(*carts))
        bounding_box_terms, dot_product_terms = zip(*terms)
        return and_(*bounding_box_terms, sum(dot_product_terms) >= cos_radius)

    @declared_attr
    def __table_args__(cls):
        *args, kwargs = super().__table_args__
        index = Index(f'ix_{cls.__tablename__}_point', *cls.cartesian)
        return (*args, index, kwargs)
